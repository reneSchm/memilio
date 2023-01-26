/* 
* Copyright (C) 2020-2023 German Aerospace Center (DLR-SC)
*
* Authors: René Schmieding
*
* Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/
#ifndef MPM_PDMM_H_
#define MPM_PDMM_H_

#include "mpm/model.h"
#include "mpm/utility.h"

#include "memilio/compartments/simulation.h"

#include <algorithm>

namespace mio
{
namespace mpm
{

template <size_t mps, class Status>
class PDMModel : public MetapopulationModel<mps, Status>
{
public:
    using MetapopulationModel<mps, Status>::MetapopulationModel;
};

} // namespace mpm

template <class Status, size_t mps>
class Simulation<mpm::PDMModel<mps, Status>>
{
    using Sim        = mio::Simulation<mpm::MetapopulationModel<mps, Status>>;
    using Integrator = mio::ControlledStepperWrapper<boost::numeric::odeint::runge_kutta_cash_karp54>;

public:
    using Model = mpm::PDMModel<mps, Status>;

    /**
     * @brief setup the simulation with an ODE solver
     * @param[in] model: An instance of a compartmental model
     * @param[in] t0 start time
     * @param[in] dt initial step size of integration
     */
    Simulation(Model const& model, ScalarType t0 = 0., ScalarType dt = 0.1)
        : m_dt(dt)
        , m_model(std::make_unique<Model>(model))
        , m_simulation(Sim(model, t0, dt))
    {
        assert(dt > 0);
        assert(std::all_of(transition_rates().begin(), transition_rates().end(), [](auto&& r) {
            return static_cast<size_t>(r.from) < mps && static_cast<size_t>(r.to) < mps;
        }));

        m_simulation.set_integrator(std::make_shared<mpm::dt_tracer>(std::make_shared<Integrator>()));
    }

    /**
     * @brief advance simulation to tmax
     * tmax must be greater than get_result().get_last_time_point()
     * @param tmax next stopping point of simulation
     */
    Eigen::Ref<Eigen::VectorXd> advance(ScalarType tmax)
    {
        // determine how long we wait until the next transition occurs
        ScalarType waiting_time = draw_waiting_time();
        ScalarType current_time = get_result().get_last_time();
        ScalarType remaining_time;
        // iterate time by increments of m_dt
        while (current_time < tmax) {
            remaining_time = std::min({m_dt, tmax - current_time});
            ; // xi * Delta-t
            // check if one or more transitions occur during this time step
            if (waiting_time < remaining_time) { // (at least one) event occurs
                std::vector<ScalarType> rates(transition_rates().size()); // lambda_m (for all m)
                // perform transition(s)
                do {
                    current_time += waiting_time; // event time t**
                    // advance all locations
                    m_simulation.advance(current_time);
                    // compute current transition rates
                    std::transform(transition_rates().begin(), transition_rates().end(), rates.begin(),
                                   [&](auto&& rate) {
                                       // we should normalize each term by dividing by the cumulative transition rate,
                                       // but the DiscreteDistribution below effectively does that for us
                                       return m_model->evaluate(rate, get_result().get_last_value());
                                   });
                    // draw transition event, then execute it
                    size_t event = mio::DiscreteDistribution<size_t>::get_instance()(rates);
                    perform_transition(event);
                    // draw new waiting time
                    remaining_time -= waiting_time;
                    waiting_time = draw_waiting_time();
                    // repeat, if another event occurs in the remaining time interval
                } while (waiting_time < remaining_time);
            }
            else { // no event occurs
                // reduce waiting time for the next step
                waiting_time -= remaining_time;
            }
            // advance time
            current_time += remaining_time;
            // advance all locations
            m_simulation.advance(current_time);
            m_dt = get_dt_tracer().get_dt();
        }
        return get_result().get_last_value();
    }

    /**
     * @brief set the core integrator used in the simulation
     */
    void set_integrator(std::shared_ptr<IntegratorCore> integrator)
    {
        get_dt_tracer().set_integrator(integrator);
    }

    /**
     * @brief get_integrator
     * @return reference to the core integrator used in the simulation
     */
    IntegratorCore& get_integrator()
    {
        return get_dt_tracer().get_integrator();
    }

    /**
     * @brief get_integrator
     * @return reference to the core integrator used in the simulation
     */
    IntegratorCore const& get_integrator() const
    {
        return get_dt_tracer().get_integrator();
    }

    /**
     * @brief get_result returns the final simulation result
     * @return a TimeSeries to represent the final simulation result
     */
    TimeSeries<ScalarType>& get_result()
    {
        return m_simulation.get_result();
    }

    /**
     * @brief get_result returns the final simulation result
     * @return a TimeSeries to represent the final simulation result
     */
    const TimeSeries<ScalarType>& get_result() const
    {
        return m_simulation.get_result();
    }

    /**
     * @brief returns the simulation model used in simulation
     */
    const Model& get_model() const
    {
        return static_cast<Model&>(m_simulation.get_model());
    }

    /**
     * @brief returns the simulation model used in simulation
     */
    Model& get_model()
    {
        return static_cast<Model&>(m_simulation.get_model());
    }

private:
    /// @brief draw time until the next event occurs
    inline ScalarType draw_waiting_time()
    {
        // compute the current cumulative transition rate
        ScalarType ctr = 0; // Lambda
        for (auto rate : transition_rates()) {
            ctr += m_model->evaluate(rate, get_result().get_last_value());
        }
        // draw the normalized waiting time
        ScalarType nwt = mio::ExponentialDistribution<ScalarType>::get_instance()(1.0); // tau'
        // "un-normalize", i.e. scale nwt by time = 1/rate
        return nwt / ctr;
    }
    /// @brief perform the given transition event, moving a single "agent"
    inline void perform_transition(size_t event)
    {
        // get the rate corresponding to the event
        const auto& rate = transition_rates()[event];
        // transitioning one person at the current time
        auto& value = get_result().get_last_value()[m_model->populations.get_flat_index({rate.from, rate.status})];
        value -= 1;
        if (value < 0) {
            std::cerr << "Transition from " << static_cast<size_t>(rate.from) << " to " << static_cast<size_t>(rate.to)
                      << " with status " << static_cast<size_t>(rate.status) << " caused negative value.\n";
            mio::log_error("Transition from {} to {} with status {} caused negative value.",
                           static_cast<size_t>(rate.from), static_cast<size_t>(rate.to),
                           static_cast<size_t>(rate.status));
        }
        get_result().get_last_value()[m_model->populations.get_flat_index({rate.to, rate.status})] += 1;
    }
    inline constexpr const typename mpm::TransitionRates<Status>::Type& transition_rates()
    {
        return m_model->parameters.template get<mpm::TransitionRates<Status>>();
    }
    // retrieve dt_tracer& from m_simulation
    inline constexpr mpm::dt_tracer& get_dt_tracer()
    {
        return static_cast<mpm::dt_tracer&>(m_simulation.get_integrator());
    }

    ScalarType m_dt;
    std::unique_ptr<const Model> m_model;
    Sim m_simulation;
};

} // namespace mio

#endif