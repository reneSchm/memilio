#include "epidemiology/abm/location.h"
#include "epidemiology/abm/person.h"
#include "epidemiology/abm/random_number_generator.h"

#include <numeric>

namespace epi
{

Location::Location(LocationType type)
    : m_type(type)
    , m_subpopulations{}
{
}

namespace
{
    /**
     * select a random transition from a list of possible transitions.
     * each transition is represented by the new state and the probability of the transition.
     * it's also possible that no transition happens in this time step.
     * in this case the old state is returned.
     * input rates don't need to sum up to probability 1, the function performs normalisation.
     * @param old_state previous state
     * @param dt length of the time step
     * @param transitions array of pairs of new states and their rates (probabilities)
     * @return new state if transition happens, old_state otherwise
     */
    template <int NumTransitions>
    InfectionState random_transition(InfectionState old_state, double dt,
                                     const std::pair<InfectionState, double> (&transitions)[NumTransitions])
    {
        auto sum = std::accumulate(std::begin(transitions), std::end(transitions), 0.0, [](auto&& sum, auto&& t) {
            return t.second + sum;
        });

        if (sum <= 0) {
            return old_state;
        }

        //time between transitions is exponentially distributed
        auto v = ExponentialDistribution<double>::get_instance()(sum);
        if (v < dt) {
            //pick a random transition
            std::array<double, NumTransitions> rates;
            std::transform(std::begin(transitions), std::end(transitions), rates.begin(), [](auto&& t) { return t.second; });
            auto random_idx = DiscreteDistribution<size_t>::get_instance()(rates);
            return transitions[random_idx].first;
        }

        return old_state;
    }
} // namespace

InfectionState Location::interact(const Person& person, double dt, const GlobalInfectionParameters& global_params) const
{
    auto state = person.get_infection_state();
    switch (state) {
    case InfectionState::Susceptible:
        return random_transition(state, dt, {{InfectionState::Exposed, dt * m_cached_exposure_rate}});
    case InfectionState::Carrier:
        return random_transition(
            state, dt,
            {{InfectionState::Infected_Detected, global_params.detect_infection * global_params.carrier_to_infected},
             {InfectionState::Infected_Undetected,
              (1 - global_params.detect_infection) * global_params.carrier_to_infected},
             {InfectionState::Recovered_Carrier, global_params.carrier_to_recovered}});
    case InfectionState::Infected_Detected: //fallthrough!
    case InfectionState::Infected_Undetected:
        return random_transition(state, dt,
                                 {{InfectionState::Recovered_Infected, global_params.infected_to_recovered},
                                  {InfectionState::Dead, global_params.infected_to_dead * m_parameters.death_factor}});
    case InfectionState::Recovered_Carrier: //fallthrough!
    case InfectionState::Recovered_Infected:
        return random_transition(state, dt, {{InfectionState::Susceptible, global_params.recovered_to_susceptible}});
    default:
        return state; //some states don't transition
    }
}

void Location::begin_step(double dt, const GlobalInfectionParameters& global_params)
{
    //cache for next step so it stays constant during the step while subpopulations change
    //otherwise we would have to cache all state changes during a step which uses more memory
    auto num_carriers = get_subpopulation(InfectionState::Carrier);
    auto num_infected =
        get_subpopulation(InfectionState::Infected_Detected) + get_subpopulation(InfectionState::Infected_Undetected);
    m_cached_exposure_rate = std::min(m_parameters.effective_contacts, double(m_num_persons)) / m_num_persons *
                             (global_params.susceptible_to_exposed_by_carrier * num_carriers +
                              global_params.susceptible_to_exposed_by_infected * num_infected);
}

void Location::add_person(const Person& p)
{
    ++m_num_persons;
    InfectionState s = p.get_infection_state();
    change_subpopulation(s, +1);
}

void Location::remove_person(const Person& p)
{
    --m_num_persons;
    InfectionState s = p.get_infection_state();
    change_subpopulation(s, -1);
}

void Location::changed_state(const Person& p, InfectionState old_state)
{
    change_subpopulation(old_state, -1);
    change_subpopulation(p.get_infection_state(), +1);
}

void Location::change_subpopulation(InfectionState s, int delta)
{
    m_subpopulations[size_t(s)] += delta;
}

int Location::get_subpopulation(InfectionState s) const
{
    return m_subpopulations[size_t(s)];
}

Eigen::Ref<const Eigen::VectorXi> Location::get_subpopulations() const
{
    return Eigen::Map<const Eigen::VectorXi>(m_subpopulations.data(), m_subpopulations.size());
}
} // namespace epi