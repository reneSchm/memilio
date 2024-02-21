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

#ifndef MPM_MODEL_H_
#define MPM_MODEL_H_

#include "mpm/region.h"

#include "memilio/epidemiology/populations.h"
#include "memilio/config.h"
#include "memilio/compartments/compartmentalmodel.h"
#include "memilio/utils/parameter_set.h"
#include "memilio/utils/time_series.h"
#include "memilio/math/stepper_wrapper.h"

#include <vector>

namespace mio
{
namespace mpm
{

// the AdoptionRate is considered to be of second order, if there are any "influences" with corresponding "factors".
// "from" is always an influence, scaled by "factor".
template <class Status>
struct AdoptionRate {
    Status from; // i
    Status to; // j
    Region region; // k
    ScalarType factor; // gammahat_{ij}^k
    std::vector<Status> influences;
    std::vector<ScalarType> factors;
};

template <class Status>
struct AdoptionRates {
    using Type = std::vector<AdoptionRate<Status>>;
    const static std::string name()
    {
        return "AdoptionRates";
    }
};

template <class Status>
struct TransitionRate {
    Status status; // i
    Region from; // k
    Region to; // l
    ScalarType factor; // lambda_i^{kl}
};

template <class Status>
struct TransitionRates {
    using Type = std::vector<TransitionRate<Status>>;
    const static std::string name()
    {
        return "TransitionRates";
    }
};

template <class Status>
using ParametersBase = mio::ParameterSet<AdoptionRates<Status>, TransitionRates<Status>>;

template <template <class State = Eigen::VectorXd, class Value = double, class Deriv = State, class Time = double,
                    class Algebra    = boost::numeric::odeint::vector_space_algebra,
                    class Operations = typename boost::numeric::odeint::operations_dispatcher<State>::operations_type,
                    class Resizer    = boost::numeric::odeint::never_resizer>
          class ControlledStepper = boost::numeric::odeint::runge_kutta_cash_karp54>
boost::numeric::odeint::controlled_runge_kutta<ControlledStepper<>> create_stepper()
{
    // for more options see: boost/boost/numeric/odeint/stepper/controlled_runge_kutta.hpp
    return boost::numeric::odeint::controlled_runge_kutta<ControlledStepper<>>(
        boost::numeric::odeint::default_error_checker<typename ControlledStepper<>::value_type,
                                                      typename ControlledStepper<>::algebra_type,
                                                      typename ControlledStepper<>::operations_type>(1e-10, 1e-5),
        boost::numeric::odeint::default_step_adjuster<typename ControlledStepper<>::value_type,
                                                      typename ControlledStepper<>::time_type>(
            std::numeric_limits<double>::max()));
}

template <size_t regions, class Status>
class MetapopulationModel
    : public mio::CompartmentalModel<Status, mio::Populations<Region, Status>, ParametersBase<Status>>
{
    using Base = mio::CompartmentalModel<Status, mio::Populations<Region, Status>, ParametersBase<Status>>;

public:
    MetapopulationModel()
        : Base(typename Base::Populations({static_cast<Region>(regions), Status::Count}, 0.),
               typename Base::ParameterSet())
        , m_number_transitions(static_cast<size_t>(Status::Count), Eigen::MatrixXd::Zero(regions, regions))
    {
    }

    mutable int eval_ctr =
        0; // keep track of runge-kutta evaluations of f' (there are 6, and the 5th is at the (potentially) next timepoint)
    const int max_evals = 6;
    mutable Eigen::MatrixXd flows; // current flow values
    mutable Eigen::VectorXd flow_x;
    mutable ScalarType flow_t, flow_dt, flow_t_old; // dt and time at which flows were recorded
    mutable decltype(create_stepper()) stepper = create_stepper();
    mutable std::shared_ptr<TimeSeries<double>> all_flows;

    void get_derivatives(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> /*pop*/, ScalarType t,
                         Eigen::Ref<Eigen::VectorXd> dxdt) const override
    {
        const auto& params = this->parameters;
        const auto& pop    = this->populations;
        if (flows.cols() != params.template get<AdoptionRates<Status>>().size()) {
            flows.resize(max_evals, params.template get<AdoptionRates<Status>>().size());
            flows.setZero();
            flow_x = Eigen::VectorXd::Zero(flows.cols());
            all_flows.reset(new TimeSeries<double>(flows.cols()));
        }
        eval_ctr++;
        if (eval_ctr == 1) {
            flow_dt = -t;
        }
        if (eval_ctr == max_evals - 1) {
            flow_t = t;
            flow_dt += t;
        }
        // evaluate each adoption rate and apply that value to its source/target
        int ctr = 0;
        for (const auto& rate : params.template get<AdoptionRates<Status>>()) {
            const auto source = pop.get_flat_index({rate.region, rate.from});
            const auto target = pop.get_flat_index({rate.region, rate.to});
            const auto change = evaluate(rate, x);
            flows(eval_ctr - 1, ctr++) += change;
            dxdt[source] -= change;
            dxdt[target] += change;
        }

        if (params.template get<AdoptionRates<Status>>().size() == 0) {
            eval_ctr = 0;
        }
        else if (eval_ctr == max_evals) {
            eval_ctr = 0;
            flow_x.setZero();
            stepper.stepper().do_step(
                [&](auto&&, Eigen::VectorXd& y, auto&&) {
                    y = flows.row(eval_ctr++);
                },
                flow_x, flow_t, flow_dt);
            eval_ctr = 0;
            flows.setZero();
            if (all_flows->get_num_time_points() == 0 || flow_t_old + flow_dt == flow_t) {
                all_flows->add_time_point(flow_t, flow_x);
            }
            else {
                all_flows->get_last_value() = flow_x;
            }
            flow_t_old = flow_t;
        }
    }

    ScalarType evaluate(const AdoptionRate<Status>& rate, const Eigen::VectorXd& x) const
    {
        assert(rate.influences.size() == rate.factors.size());
        const auto& pop   = this->populations;
        const auto source = pop.get_flat_index({rate.region, rate.from});
        // determine order and calculate rate
        if (rate.influences.size() == 0) { // first order adoption
            return rate.factor * x[source];
        }
        else { // second order adoption
            ScalarType N = 0;
            for (size_t s = 0; s < static_cast<size_t>(Status::Count); ++s) {
                N += x[pop.get_flat_index({rate.region, Status(s)})];
            }
            // accumulate influences
            ScalarType influences = 0.0;
            for (size_t i = 0; i < rate.influences.size(); i++) {
                influences += rate.factors[i] * x[pop.get_flat_index({rate.region, rate.influences[i]})];
            }
            return (N > 0) ? (rate.factor * x[source] * influences / N) : 0;
        }
    }

    ScalarType evaluate(const TransitionRate<Status>& rate, const Eigen::VectorXd& x) const
    {
        const auto source = this->populations.get_flat_index({rate.from, rate.status});
        return rate.factor * x[source];
    }
    double& number_transitions(const mpm::TransitionRate<Status>& tr) const
    {
        return m_number_transitions[static_cast<size_t>(tr.status)](static_cast<size_t>(tr.from),
                                                                    static_cast<size_t>(tr.to));
    }

    const std::vector<Eigen::MatrixXd>& number_transitions() const
    {
        return m_number_transitions;
    }

    void increase_number_transitions(const mpm::TransitionRate<Status>& tr) const
    {
        m_number_transitions[static_cast<size_t>(tr.status)](static_cast<size_t>(tr.from),
                                                             static_cast<size_t>(tr.to))++;
    }

private:
    mutable std::vector<Eigen::MatrixXd> m_number_transitions;
};

} // namespace mpm
} // namespace mio

#endif