#ifndef ABM_ADAPTER_H_
#define ABM_ADAPTER_H_

#include "models/abm/world.h"
#include "models/abm/simulation.h"
#include "memilio/compartments/simulation.h"

namespace mio
{

// a specialization of mio::Simulation (usually for compartmental models) to suport abm::World
// this class is used to run hybrid simulations
template <>
class Simulation<abm::World> : public abm::Simulation
{
public:
    using abm::Simulation::Simulation;
    void advance(double tmax)
    {
        abm::Simulation::advance(abm::TimePoint(tmax * (24 * 60 * 60)));
    }
    abm::World& get_model()
    {
        return abm::Simulation::get_world();
    }
    const abm::World& get_model() const
    {
        return abm::Simulation::get_world();
    }
};

template <class Model>
Simulation<Model> create_simulation(Model& m, double& t0, double& dt);

template <>
inline Simulation<abm::World> create_simulation(abm::World& m, double& t0, double& dt)
{
    return Simulation<abm::World>(abm::TimePoint(t0 * (24 * 60 * 60)), std::move(m));
}

} // namespace mio

#endif