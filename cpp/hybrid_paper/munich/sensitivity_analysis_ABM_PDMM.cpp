#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/munich/munich_setup.h"
#include "hybrid_paper/munich/sensitivity_analysis_setup_munich.h"
#include "memilio/utils/logging.h"
#include "hybrid_paper/library/infection_state.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "hybrid_paper/library/sensitivity_analysis.h"
#include "sensitivity_analysis_setup_munich.h"
#include "munich_setup.h"
#include <cstddef>
#include <omp.h>

int main()
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status                     = mio::mpm::paper::InfectionState;
    using ABM                        = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    const size_t num_regions         = 8;
    using PDMM                       = mio::mpm::PDMModel<num_regions, Status>;
    const size_t num_runs            = 10;
    const size_t num_runs_per_output = 90;
    const size_t num_agents          = 4000;
    double tmax                      = 150;
    double dt                        = 0.1;
    std::cout << "num runs: " << num_runs << std::endl;
    std::cout << "num agents: " << num_agents << std::endl;

    std::string result_dir = mio::base_dir() + "cpp/outputs/sensitivity_analysis/20241009_v2/";

    SensitivitySetupMunich sensi_setup(num_runs, 4);
    auto draw_func_abm = [](mio::mpm::paper::MunichSetup<ABM::Agent> setup, auto& sim) {
        setup.redraw_agents_status(sim);
    };
    auto draw_func_pdmm = [](mio::mpm::paper::MunichSetup<ABM::Agent> setup, auto& sim) {
        setup.redraw_pdmm_populations(sim);
    };

    const auto& output_func_abm =
        sensitivity_results<mio::mpm::paper::MunichSetup<ABM::Agent>, ABM, decltype(draw_func_abm)>;
    const auto& output_func_pdmm =
        sensitivity_results<mio::mpm::paper::MunichSetup<ABM::Agent>, PDMM, decltype(draw_func_pdmm)>;

    run_sensitivity_analysis<SensitivitySetupMunich, ABM, mio::mpm::paper::MunichSetup<ABM::Agent>,
                             decltype(output_func_abm), decltype(draw_func_abm)>(
        sensi_setup, output_func_abm, num_runs, num_agents, tmax, dt, result_dir + "ABM", num_runs_per_output,
        draw_func_abm);
    run_sensitivity_analysis<SensitivitySetupMunich, PDMM, mio::mpm::paper::MunichSetup<ABM::Agent>,
                             decltype(output_func_pdmm), decltype(draw_func_pdmm)>(
        sensi_setup, output_func_pdmm, num_runs, num_agents, tmax, dt, result_dir + "PDMM", num_runs_per_output,
        draw_func_pdmm);

    return 0;
}
