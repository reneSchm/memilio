#include "infection_state.h"

namespace mio
{

namespace mpm
{

namespace paper
{

Eigen::Index get_region_flow_index(int region, InfectionState from, InfectionState to)
{
    return Eigen::Index(flow_indices.size() * region + flow_indices.at({from, to}));
}

} //namespace paper
} // namespace mpm
} // namespace mio