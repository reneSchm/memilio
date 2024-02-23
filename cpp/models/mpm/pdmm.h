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

#include "memilio/config.h"
#include "mpm/model.h"
#include "mpm/utility.h"

#include "memilio/compartments/simulation.h"

#include <algorithm>
#include <numeric>

namespace mio
{
namespace mpm
{

template <size_t regions, class Status>
class PDMModel : public MetapopulationModel<regions, Status>
{
public:
    using MetapopulationModel<regions, Status>::MetapopulationModel;
};

} // namespace mpm

template <class Status, size_t regions>
class Simulation<mpm::PDMModel<regions, Status>>
{
    using Sim        = mio::Simulation<mpm::MetapopulationModel<regions, Status>>;
    using Integrator = mio::ControlledStepperWrapper<boost::numeric::odeint::runge_kutta_cash_karp54>;

public:
    using Model = mpm::PDMModel<regions, Status>;

    /**
     * @brief setup the simulation with an ODE solver
     * @param[in] model: An instance of a compartmental model
     * @param[in] t0 start time
     * @param[in] dt initial step size of integration
     */
    Simulation(Model const& model, ScalarType t0 = 0., ScalarType dt = 0.1)
        : m_dt(dt)
        , m_normalized_waiting_time(mio::ExponentialDistribution<ScalarType>::get_instance()(1.0))
        , m_rates(model.parameters.template get<mpm::TransitionRates<Status>>().size())
        , m_simulation(Sim(model, t0, dt))
    {
        assert(dt > 0);
        assert(std::all_of(transition_rates().begin(), transition_rates().end(), [](auto&& r) {
            return static_cast<size_t>(r.from) < regions && static_cast<size_t>(r.to) < regions;
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
        ScalarType current_time = get_result().get_last_time();
        ScalarType remaining_time; // xi * Delta-t
        ScalarType cctr = 0; // Lambda (aka current cumulative transition rate)
        // iterate time by increments of m_dt
        while (current_time < tmax) {
            remaining_time = std::min({m_dt, tmax - current_time});
            // update current (cumulative) transition rates
            evaluate_current_tramsition_rates(m_rates, cctr);
            // check if one or more transitions occur during this time step
            if (m_normalized_waiting_time < remaining_time * cctr) { // (at least one) event occurs
                // perform transition(s)
                do {
                    current_time += m_normalized_waiting_time / cctr; // event time t**
                    // advance all locations to t**
                    m_simulation.advance(current_time);
                    // draw which transition event occurs, then execute it
                    size_t event = mio::DiscreteDistribution<size_t>::get_instance()(m_rates);
                    perform_transition(event);
                    remaining_time -= m_normalized_waiting_time / cctr;
                    // update current (cumulative) transition rates
                    evaluate_current_tramsition_rates(m_rates, cctr);
                    // draw new waiting time
                    m_normalized_waiting_time = mio::ExponentialDistribution<ScalarType>::get_instance()(1.0);
                    // repeat, if another event occurs in the remaining time interval
                } while (m_normalized_waiting_time < remaining_time * cctr);
            }
            else { // no event occurs
                // reduce waiting time for the next step
                m_normalized_waiting_time -= remaining_time * cctr;
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
    /// @brief compute current transition rates and the accumulated transiton rate
    inline void evaluate_current_tramsition_rates(std::vector<ScalarType>& rates, ScalarType& cctr)
    {
        std::transform(transition_rates().begin(), transition_rates().end(), rates.begin(), [&](auto&& r) {
            // we should normalize each term by dividing by the cumulative transition rate,
            // but the DiscreteDistribution used for drawing "event" effectively does that for us
            const ScalarType rate = get_model().evaluate(r, get_result().get_last_value());
            const ScalarType from =
                get_result().get_last_value()[get_model().populations.get_flat_index({r.from, r.status})];
            // clamp rates to 0, if the adoption event would cause a negative value by moving 1 agent
            return from >= 1 ? rate : 0;
        });
        cctr = std::accumulate(rates.begin(), rates.end(), 0.0);
    }
    /// @brief perform the given transition event, moving a single "agent"
    inline void perform_transition(size_t event)
    {
        // get the rate corresponding to the event
        const auto& rate = transition_rates()[event];
        get_model().increase_number_transitions(rate);
        // transitioning one person at the current time
        get_result().get_last_value()[get_model().populations.get_flat_index({rate.from, rate.status})] -= 1;
        get_result().get_last_value()[get_model().populations.get_flat_index({rate.to, rate.status})] += 1;
    }
    inline constexpr const typename mpm::TransitionRates<Status>::Type& transition_rates()
    {
        return get_model().parameters.template get<mpm::TransitionRates<Status>>();
    }
    // retrieve dt_tracer& from m_simulation
    inline constexpr mpm::dt_tracer& get_dt_tracer()
    {
        return static_cast<mpm::dt_tracer&>(m_simulation.get_integrator());
    }

    ScalarType m_dt;
    ScalarType m_normalized_waiting_time; // tau'
    std::vector<ScalarType> m_rates; // lambda_m (for all m)
    Sim m_simulation;
};

} // namespace mio

#endif