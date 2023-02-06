#ifndef HYBRID_H_
#define HYBRID_H_

#include "abm_adapter.h"
#include "memilio/math/eigen.h"
#include "memilio/compartments/simulation.h"

#include <cassert>
#include <vector>

//TODO: remove DEBUG
//#include <iostream>
#define DEBUG(cout_args) std::cerr << cout_args << std::endl << std::flush;
//#define DEBUG(cout_args) (void)0;

namespace mio
{

template <class BaseModel, class TargetModel>
const TimeSeries<ScalarType>::Vector convert_result_values(const TimeSeries<ScalarType>::Vector& value)
{
    DEBUG("convert_result_values default")
    return value;
}

template <class BaseModel, class TargetModel>
void convert_model(const Simulation<BaseModel>&, TargetModel&) = delete;
//    static_assert(sizeof(BaseModel) == 0, "convert_model requires specialization!");

template <class Model>
Simulation<Model> create_simulation(Model& m, double& t0, double& dt)
{
    return Simulation<Model>(m, t0, dt);
}

template <class BaseModel, class SecondaryModel>
class HybridSimulation
{

public:
    using switch_fct =
        std::function<bool(const bool& using_base, const double&, const BaseModel&, const SecondaryModel&)>;

    HybridSimulation(BaseModel& base_model, SecondaryModel& secondary_model, double dt_switch, double t0 = 0,
                     double dt = 0.1)
        : m_dt(dt)
        , m_dt_switch(dt_switch)
        , m_sim_base(std::move(create_simulation(base_model, t0, dt)))
        , m_sim_sec(std::move(create_simulation(secondary_model, t0, dt)))
        , m_results(t0, m_sim_base.get_result().get_value(0))
    {
        DEBUG("HybridSimulation")
        m_sim_sec.get_result().remove_time_point(0);
    }

    void advance(double tmax, const switch_fct& use_base_model)
    {
        DEBUG("advance")
        assert(tmax > 0);
        double t = get_result().get_last_time(); // current hybrid sim time
        while (t < tmax) {
            DEBUG(t << "\t/ " << std::min(t + m_dt_switch, tmax) << "\t/ " << tmax << " :: " << m_using_base_model
                    << " " << use_base_model(m_using_base_model, t, m_sim_base.get_model(), m_sim_sec.get_model()))
            if (m_using_base_model !=
                use_base_model(m_using_base_model, t, m_sim_base.get_model(), m_sim_sec.get_model())) {
                if (m_using_base_model) {
                    m_using_base_model = false;
                    // set up sec model to start at the current state of the hybrid simulation
                    convert_model<BaseModel, SecondaryModel>(m_sim_base, m_sim_sec.get_model());
                    // save the results of base simulation to m_result, then clear base sim's result
                    move_result<BaseModel>(m_sim_base.get_result());
                    // set the start point for sec simulation
                    set_start_point<SecondaryModel>(m_sim_sec.get_result());
                }
                else {
                    m_using_base_model = true;
                    // set up base model to start at the current state of the hybrid simulation
                    convert_model<SecondaryModel, BaseModel>(m_sim_sec, m_sim_base.get_model());
                    // save and convert the results of sec simulation to m_result, then clear sec sim's result
                    move_result<SecondaryModel>(m_sim_sec.get_result());
                    // set the start point for base simulation
                    set_start_point<BaseModel>(m_sim_base.get_result());
                }
            }
            double t_step = std::min(t + m_dt_switch, tmax);
            if (m_using_base_model) {
                m_sim_base.advance(t_step);
            }
            else {
                m_sim_sec.advance(t_step);
            }
            t = t_step;
        }
        // save results from last step to m_result
        if (m_using_base_model) {
            // save the results of simulation A to m_result, then clear sim A's result
            move_result<BaseModel>(m_sim_base.get_result());
            // re-add the last time point to simulation A
            set_start_point<BaseModel>(m_sim_base.get_result());
        }
        else {
            // save the results of simulation B to m_result, then clear sim B's result
            move_result<SecondaryModel>(m_sim_sec.get_result());
            // re-add the last time point to simulation B
            // TODO ? change this fct, as may loose information here (depending on convert_result_values implementation)
            set_start_point<SecondaryModel>(m_sim_sec.get_result());
        }
    }

    /**
     * @brief get_result returns the final simulation result
     * @return a TimeSeries to represent the final simulation result
     */
    const TimeSeries<ScalarType>& get_result() const
    {
        DEBUG("get_result")
        return m_results;
    }

private:
    template <class Model>
    void move_result(TimeSeries<ScalarType>& ts)
    {
        DEBUG("move_result")
        const auto end = ts.get_num_time_points();
        for (Eigen::Index i = 1; i < end; i++) {
            m_results.add_time_point(ts.get_time(i), convert_result_values<Model, BaseModel>(ts.get_value(i)));
        }
        for (Eigen::Index i = 0; i < end; ++i) {
            ts.remove_last_time_point();
        }
    }
    template <class Model>
    void set_start_point(TimeSeries<ScalarType>& ts)
    {
        DEBUG("set_start_point")
        assert(ts.get_num_time_points() == 0);
        ts.add_time_point(m_results.get_last_time(),
                          convert_result_values<BaseModel, Model>(m_results.get_last_value()));
    }

    double m_dt, m_dt_switch;
    bool m_using_base_model = true;
    Simulation<BaseModel> m_sim_base;
    Simulation<SecondaryModel> m_sim_sec;
    TimeSeries<ScalarType> m_results;
};

} // namespace mio

#endif // HYBRID_H_