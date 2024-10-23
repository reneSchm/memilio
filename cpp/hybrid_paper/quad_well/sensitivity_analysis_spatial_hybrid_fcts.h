#include "hybrid_paper/library/quad_well.h"
#include "hybrid_paper/quad_well/quad_well_setup.h"
#include "sensitivity_analysis_setup_qw.h"
#include "hybrid_paper/library/sensitivity_analysis.h"

void set_up_models(mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>& abm,
                   mio::mpm::PDMModel<4, mio::mpm::paper::InfectionState>& pdmm);

std::vector<double>
simulate_hybridization(mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>& abm,
                       mio::mpm::PDMModel<4, mio::mpm::paper::InfectionState>& pdmm,
                       const QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>& setup,
                       size_t num_runs);

void run_sensitivity_analysis_hybrid(SensitivitySetupQW& sensi_setup, size_t num_runs, size_t num_runs_per_output,
                                     size_t num_agents, double tmax, double dt, std::string result_dir);
