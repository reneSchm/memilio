#ifndef TRANSITION_RATE_ESTIMATION_H
#define TRANSITION_RATE_ESTIMATION_H

#include "hybrid_paper/library/infection_state.h"
#include "memilio/compartments/simulation.h"
#include "mpm/model.h"

namespace mio
{

namespace mpm
{

namespace paper
{

std::vector<TransitionRate<InfectionState>> add_transition_rates(std::vector<TransitionRate<InfectionState>>& v1,
                                                                 std::vector<TransitionRate<InfectionState>>& v2);

void print_transition_rates(std::vector<TransitionRate<InfectionState>>& transition_rates, bool print_to_file);

template <class ABM>
std::vector<TransitionRate<InfectionState>> calculate_transition_rates(ABM& abm, size_t num_runs, double tmax,
                                                                       size_t num_regions,
                                                                       std::vector<double>& ref_pops, double ppa)
{
    std::vector<std::vector<TransitionRate<InfectionState>>> estimated_transition_rates(num_runs);

    std::vector<TransitionRate<InfectionState>> zero_transition_rates;

    for (size_t i = 0; i < num_regions; ++i) {
        for (size_t j = 0; j < num_regions; ++j) {
            if (i != j) {
                zero_transition_rates.push_back({InfectionState::S, mio::mpm::Region(i), mio::mpm::Region(j), 0});
                zero_transition_rates.push_back({InfectionState::E, mio::mpm::Region(i), mio::mpm::Region(j), 0});
                zero_transition_rates.push_back({InfectionState::C, mio::mpm::Region(i), mio::mpm::Region(j), 0});
                zero_transition_rates.push_back({InfectionState::I, mio::mpm::Region(i), mio::mpm::Region(j), 0});
                zero_transition_rates.push_back({InfectionState::R, mio::mpm::Region(i), mio::mpm::Region(j), 0});
                zero_transition_rates.push_back({InfectionState::D, mio::mpm::Region(i), mio::mpm::Region(j), 0});
            }
        }
    }
    std::fill_n(estimated_transition_rates.begin(), num_runs, zero_transition_rates);
    for (size_t run = 0; run < num_runs; ++run) {
        std::cerr << "run number: " << run << "\n" << std::flush;
        mio::Simulation<ABM> sim(abm, 0, 0.05);
        sim.advance(tmax);

        //add calculated transition rates
        for (auto& tr_rate : estimated_transition_rates[run]) {
            auto region_pop = ref_pops[tr_rate.from.get()] / ppa;
            tr_rate.factor  = sim.get_model().number_transitions(tr_rate) / (region_pop * tmax);
        }
    }

    auto mean_transition_rates = std::accumulate(estimated_transition_rates.begin(), estimated_transition_rates.end(),
                                                 zero_transition_rates, add_transition_rates);

    double denominator{(1 / (double)num_runs)};
    std::transform(
        mean_transition_rates.begin(), mean_transition_rates.end(), mean_transition_rates.begin(),
        [&denominator](mio::mpm::TransitionRate<InfectionState>& rate) {
            return mio::mpm::TransitionRate<InfectionState>{rate.status, rate.from, rate.to, denominator * rate.factor};
        });

    print_transition_rates(mean_transition_rates, true);
    return mean_transition_rates;
}

} // namespace paper

} // namespace mpm

} // namespace mio

#endif //TRANSITION_RATE_ESTIMATION_H