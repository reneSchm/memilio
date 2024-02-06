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

template <size_t regions, class Status>
class MetapopulationModel
    : public mio::CompartmentalModel<Status, mio::Populations<Region, Status>, ParametersBase<Status>>
{
    using Base = mio::CompartmentalModel<Status, mio::Populations<Region, Status>, ParametersBase<Status>>;

public:
    MetapopulationModel()
        : Base(typename Base::Populations({static_cast<Region>(regions), Status::Count}, 0.),
               typename Base::ParameterSet())
    {
    }

    void get_derivatives(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd> /*pop*/,
                         ScalarType /*t*/, Eigen::Ref<Eigen::VectorXd> dxdt) const override
    {
        const auto& params = this->parameters;
        const auto& pop    = this->populations;
        for (const auto& rate : params.template get<AdoptionRates<Status>>()) {
            const auto source = pop.get_flat_index({rate.region, rate.from});
            const auto target = pop.get_flat_index({rate.region, rate.to});
            const auto change = evaluate(rate, x);
            dxdt[source] -= change;
            dxdt[target] += change;
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
            const ScalarType N = pop.get_group_total(rate.region);
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
};

} // namespace mpm
} // namespace mio

#endif