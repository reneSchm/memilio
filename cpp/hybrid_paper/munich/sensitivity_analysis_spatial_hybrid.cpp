#include "memilio/utils/logging.h"
#include "sensitivity_analysis_setup_munich.h"
#include "sensitivity_analysis_spatial_hybrid_fcts.h"

int main()
{
    mio::set_log_level(mio::LogLevel::warn);
    const size_t num_regions         = 8;
    const size_t num_runs            = 10;
    const size_t num_runs_per_output = 90;
    const size_t num_agents          = 4000;
    double tmax                      = 150.0;
    double dt                        = 0.1;
    const size_t num_outputs         = 5;

    std::string result_dir = mio::base_dir() + "cpp/outputs/sensitivity_analysis/20241011_v1/";

    SensitivitySetupMunich sensi_setup(num_runs, num_outputs);
    run_sensitivity_analysis_hybrid(sensi_setup, num_runs, num_runs_per_output, tmax, dt, num_agents,
                                    result_dir + "Hybrid");

    return 0;
}
