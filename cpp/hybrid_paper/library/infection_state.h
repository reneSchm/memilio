#ifndef MIO_MPM_INFECTIONSTATE_H_
#define MIO_MPM_INFECTIONSTATE_H_

#include "memilio/math/eigen.h"

#include <map>
namespace mio
{

namespace mpm
{

namespace paper
{
/**
 * @brief The InfectionState enum describes the possible
 * categories for the infectious state of persons
 */
enum class InfectionState
{
    S,
    E,
    C,
    I,
    R,
    D,
    Count
};

std::map<std::tuple<InfectionState, InfectionState>, Eigen::Index> flow_indices{
    {{InfectionState::S, InfectionState::E}, Eigen::Index(0)},
    {{InfectionState::E, InfectionState::C}, Eigen::Index(1)},
    {{InfectionState::C, InfectionState::I}, Eigen::Index(2)},
    {{InfectionState::C, InfectionState::R}, Eigen::Index(3)},
    {{InfectionState::I, InfectionState::R}, Eigen::Index(4)},
    {{InfectionState::I, InfectionState::D}, Eigen::Index(5)}};

Eigen::Index get_region_flow_index(int region, InfectionState from, InfectionState to)
{
    return Eigen::Index(flow_indices.size() * region + flow_indices.at({from, to}));
}
} //namespace paper
} // namespace mpm
} // namespace mio

#endif // MIO_MPM_INFECTIONSTATE_H_