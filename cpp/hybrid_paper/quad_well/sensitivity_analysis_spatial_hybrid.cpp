#include "hybrid_paper/library/quad_well.h"
#include "hybrid_paper/quad_well/quad_well_setup.h"
#include "memilio/data/analyze_result.h"
#include "memilio/utils/logging.h"
#include "memilio/utils/random_number_generator.h"
#include "memilio/utils/time_series.h"
#include "sensitivity_analysis_setup_qw.h"
#include "sensitivity_analysis_spatial_hybrid_fcts.h"
#include "hybrid_paper/library/sensitivity_analysis.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <ostream>
#include <vector>

int main()
{
    mio::set_log_level(mio::LogLevel::warn);

    const size_t num_regions         = 4;
    const size_t num_runs            = 40;
    const size_t num_runs_per_output = 112;
    const size_t num_agents          = 4000;
    double tmax                      = 150.0;
    double dt                        = 0.1;
    const size_t num_outputs         = 5;

    std::string result_dir = mio::base_dir() + "cpp/outputs/20241011_finalHybrid/";

    SensitivitySetupQW sensi_setup(num_runs, num_outputs);
    run_sensitivity_analysis_hybrid(sensi_setup, num_runs, num_runs_per_output, num_agents, tmax, dt,
                                    result_dir + "Hybrid");

    return 0;
}
