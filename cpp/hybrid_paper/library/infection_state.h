#ifndef MIO_MPM_INFECTIONSTATE_H_
#define MIO_MPM_INFECTIONSTATE_H_

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
} //namespace paper
} // namespace mpm
} // namespace mio

#endif // MIO_MPM_INFECTIONSTATE_H_