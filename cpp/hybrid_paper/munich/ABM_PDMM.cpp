#include "hybrid_paper/library/infection_state.h"
#include "models/mpm/region.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "hybrid_paper/munich/munich_setup.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/ensemble_run.h"
#include "memilio/utils/time_series.h"
#include "memilio/data/analyze_result.h"
#include <cstddef>
#include <memory>
#include <numeric>

int main()
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status = mio::mpm::paper::InfectionState;
    using Region = mio::mpm::Region;
    using ABM    = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    using PDMM   = mio::mpm::PDMModel<8, Status>;

    size_t num_runs = 3;
    mio::mpm::paper::MunichSetup<ABM::Agent> setup;

    ABM abm   = setup.create_abm<ABM>();
    PDMM pdmm = setup.create_pdmm<PDMM>();

    // std::cout << "Transitions real per day\n";
    // for (size_t i = 0; i < setup.metaregions.maxCoeff(); ++i) {
    //     for (size_t j = 0; j < setup.metaregions.maxCoeff(); ++j) {
    //         if (i != j) {
    //             std::cout << i << " -> " << j << ": "
    //                       << (setup.commute_weights(i, j) + setup.commute_weights(j, i)) / setup.persons_per_agent
    //                       << "\n";
    //         }
    //     }
    // }

    // std::cout << "Commuters real per day\n";
    // for (size_t i = 0; i < setup.metaregions.maxCoeff(); ++i) {
    //     for (size_t j = 0; j < setup.metaregions.maxCoeff(); ++j) {
    //         if (i != j) {
    //             std::cout << i << " -> " << j << ": " << (setup.commute_weights(i, j)) / setup.persons_per_agent
    //                       << "\n";
    //         }
    //     }
    // }

    // auto& transition_rates = pdmm.parameters.get<mio::mpm::TransitionRates<Status>>();
    // for (auto& rate : transition_rates) {
    //     rate.factor = (setup.commute_weights(static_cast<size_t>(rate.from), static_cast<size_t>(rate.to)) +
    //                    setup.commute_weights(static_cast<size_t>(rate.to), static_cast<size_t>(rate.from))) /
    //                   setup.populations[static_cast<size_t>(rate.from)];
    // }

    //std::cout << (setup.commute_weights.array() / setup.persons_per_agent).matrix() << "\n";
    //std::cout << "num_agents_pdmm: " << pdmm.populations.get_total() << std::endl;
    //std::cout << "num_agents_abm: " << abm.populations.size() << std::endl;
    auto draw_func_abm = [&setup](auto& model) {
        setup.redraw_agents_status(model);
    };
    auto draw_func_pdmm = [&setup](auto& model) {
        setup.redraw_pdmm_populations(model);
    };

    auto draw_func_no_draw = [&setup](auto& model) {
        setup.dummy(model);
    };
    // mio::mpm::paper::run(abm, num_runs, setup.tmax, setup.dt, setup.metaregions.maxCoeff(), false, "test_ABM",
    //                      mio::base_dir() + "cpp/outputs/Munich/test/", draw_func_abm);
    mio::mpm::paper::run(pdmm, num_runs, setup.tmax, setup.dt, setup.metaregions.maxCoeff(), false, "test_PDMM",
                         mio::base_dir() + "cpp/outputs/Munich/test/", draw_func_pdmm);

    return 0;
}