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

#include "mpm/metapopulation.h"

#include "memilio/epidemiology/populations.h"
#include "memilio/config.h"
#include "memilio/compartments/compartmentalmodel.h"
#include "memilio/utils/parameter_set.h"

#include <vector>

namespace mio
{
namespace mpm
{

enum class Order
{
    first,
    second
};

template <class Status>
struct AdoptionRate {
    Status from; // i
    Status to; // j
    ScalarType factor; // gammahat_{ij}^k
    Order order; // first or second
    Metapopulation location; // k
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
    Metapopulation from; // k
    Metapopulation to; // l
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

template <size_t mps, class Status>
class MetapopulationModel
    : public mio::CompartmentalModel<Status, mio::Populations<Metapopulation, Status>, ParametersBase<Status>>
{
    using Base = mio::CompartmentalModel<Status, mio::Populations<Metapopulation, Status>, ParametersBase<Status>>;

public:
    MetapopulationModel()
        : Base(typename Base::Populations({static_cast<Metapopulation>(mps), Status::Count}, 0.),
               typename Base::ParameterSet())
    {
    }

    void get_derivatives(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> /*pop*/,
                         ScalarType /*t*/, Eigen::Ref<Eigen::VectorXd> dxdt) const override
    {
        const auto& params = this->parameters;
        const auto& pop    = this->populations;
        for (const auto& rate : params.template get<AdoptionRates<Status>>()) {
            const auto source = pop.get_flat_index({rate.location, rate.from});
            const auto target = pop.get_flat_index({rate.location, rate.to});
            const auto change = evaluate(rate, x);
            dxdt[source] -= change;
            dxdt[target] += change;
        }
    }

    static constexpr ScalarType evaluate(const AdoptionRate<Status>& rate, const Eigen::VectorXd& x)
    {
        const auto source = flat_index({rate.location, rate.from});
        const auto target = flat_index({rate.location, rate.to});
        // calculate rate depending on order
        if (rate.order == Order::first) {
            return rate.factor * x[source];
        }
        else {
            // calculate current size of metapopulation
            const ScalarType N =
                x.segment(rate.location.get() * static_cast<size_t>(Status::Count), static_cast<size_t>(Status::Count))
                    .sum();
            return (N > 0) ? (rate.factor * x[source] * x[target] / N) : 0;
        }
    }

    static constexpr ScalarType evaluate(const TransitionRate<Status>& rate, const Eigen::VectorXd& x)
    {
        const auto source = flat_index({rate.from, rate.status});
        return rate.factor * x[source];
    }

private:
    static constexpr Eigen::Index flat_index(const typename Base::Populations::Index& indices)
    {
        return mio::details::flatten_index<0>(indices, {static_cast<Metapopulation>(mps), Status::Count}).first;
    }
};

} // namespace mpm
} // namespace mio

#endif