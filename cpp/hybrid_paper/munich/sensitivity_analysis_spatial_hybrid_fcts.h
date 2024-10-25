#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "hybrid_paper/munich/munich_setup.h"
#include "sensitivity_analysis_setup_munich.h"

void set_up_models(mio::mpm::ABM<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>>& abm,
                   mio::mpm::PDMModel<8, mio::mpm::paper::InfectionState>& pdmm);

std::vector<double> simulate_hybridization(
    mio::mpm::ABM<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>>& abm,
    mio::mpm::PDMModel<8, mio::mpm::paper::InfectionState>& pdmm,
    const mio::mpm::paper::MunichSetup<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>::Agent>& setup,
    size_t num_runs_per_output);

void run_sensitivity_analysis_hybrid(SensitivitySetupMunich& sensi_setup, size_t num_runs, size_t num_runs_per_output,
                                     double tmax, double dt, size_t num_agents, std::string result_dir);
