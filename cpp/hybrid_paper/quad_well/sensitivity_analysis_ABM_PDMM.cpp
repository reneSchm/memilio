#include "hybrid_paper/library/ensemble_run.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/sensitivity_analysis.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "mpm/region.h"
#include "quad_well_setup.h"
#include "sensitivity_analysis_setup.h"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include <omp.h>

int main()
{

    mio::set_log_level(mio::LogLevel::warn);
    using Status                     = mio::mpm::paper::InfectionState;
    using ABM                        = mio::mpm::ABM<QuadWellModel<Status>>;
    const size_t num_regions         = 4;
    using PDMM                       = mio::mpm::PDMModel<num_regions, Status>;
    const size_t num_runs            = 100;
    const size_t num_runs_per_output = 1;
    const size_t num_agents          = 8000;
    double tmax                      = 150.0;
    double dt                        = 0.1;
    std::cout << "num runs: " << num_runs << std::endl;
    std::cout << "num agents: " << num_agents << std::endl;
    mio::thread_local_rng().seed({static_cast<uint32_t>(0)});

    std::string result_dir = mio::base_dir() + "cpp/outputs/sensitivity_analysis/20240925_v1/";

    SensitivitySetup sensi_setup(num_runs, 4);
    auto draw_func_abm = [](QuadWellSetup<ABM::Agent> setup, auto& sim) {
        setup.redraw_agents_status(sim);
    };
    auto draw_func_pdmm = [](QuadWellSetup<ABM::Agent> setup, auto& sim) {
        setup.redraw_pdmm_populations(sim);
    };
    const auto& output_func_abm  = sensitivity_results<QuadWellSetup<ABM::Agent>, ABM, decltype(draw_func_abm)>;
    const auto& output_func_pdmm = sensitivity_results<QuadWellSetup<ABM::Agent>, PDMM, decltype(draw_func_pdmm)>;

    // run_sensitivity_analysis<SensitivitySetup, ABM, QuadWellSetup<ABM::Agent>, decltype(output_func_abm),
    //                          decltype(draw_func_abm)>(sensi_setup, output_func_abm, num_runs, num_agents, tmax, dt,
    //                                                   result_dir + "ABM_", num_runs_per_output, draw_func_abm);
    run_sensitivity_analysis<SensitivitySetup, PDMM, QuadWellSetup<ABM::Agent>, decltype(output_func_pdmm),
                             decltype(draw_func_pdmm)>(sensi_setup, output_func_pdmm, num_runs, num_agents, tmax, dt,
                                                       result_dir + "PDMM_", num_runs_per_output, draw_func_pdmm);

    return 0;
}
