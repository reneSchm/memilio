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

#ifndef MPM_UTILITY_H_
#define MPM_UTILITY_H_

#include "memilio/config.h"
#include "memilio/math/eigen.h"
#include "memilio/math/integrator.h"
#include "memilio/utils/time_series.h"

namespace mio
{
namespace mpm
{

void print_to_terminal(const mio::TimeSeries<ScalarType>& results, const std::vector<std::string>& state_names);

class dt_tracer : public mio::IntegratorCore
{
public:
    explicit dt_tracer(std::shared_ptr<IntegratorCore> integrator)
        : m_dt()
        , m_integrator(std::move(integrator))
    {
    }

    inline bool step(const DerivFunction& f, Eigen::Ref<const Eigen::VectorXd> yt, ScalarType& t, ScalarType& dt,
                     Eigen::Ref<Eigen::VectorXd> ytp1) const override final
    {
        const auto rtval = m_integrator->step(f, yt, t, dt, ytp1);
        m_dt             = dt;
        return rtval;
    }

    void set_integrator(std::shared_ptr<IntegratorCore> integrator)
    {
        m_integrator = std::move(integrator);
    }

    IntegratorCore& get_integrator()
    {
        return *m_integrator;
    }

    const IntegratorCore& get_integrator() const
    {
        return *m_integrator;
    }

    ScalarType get_dt() const
    {
        return m_dt;
    }

private:
    mutable ScalarType m_dt;
    std::shared_ptr<IntegratorCore> m_integrator;
};

} // namespace mpm
} // namespace mio

#endif