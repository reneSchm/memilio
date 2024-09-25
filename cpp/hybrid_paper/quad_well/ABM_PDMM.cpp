#include "hybrid_paper/library/ensemble_run.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/initialization.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "quad_well_setup.h"
#include <cstddef>

int main()
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status             = mio::mpm::paper::InfectionState;
    using ABM                = mio::mpm::ABM<QuadWellModel<Status>>;
    const size_t num_regions = 4;
    using PDMM               = mio::mpm::PDMModel<num_regions, Status>;

    size_t num_runs = 140;
    const QuadWellSetup<ABM::Agent> setup(8000);
    ABM abm            = setup.create_abm<ABM>();
    PDMM pdmm          = setup.create_pdmm<PDMM>();
    auto draw_func_abm = [&setup](auto& model) {
        setup.redraw_agents_status(model);
    };
    auto draw_func_pdmm = [&setup](auto& model) {
        setup.redraw_pdmm_populations(model);
    };

    auto draw_func_no_draw = [&setup](auto& model) {
        setup.dummy(model);
    };
    bool save_percentiles  = true;
    std::string result_dir = mio::base_dir() + "cpp/outputs/QuadWell/20240923_v1/";
    setup.save_setup(result_dir);
    mio::mpm::paper::run(abm, num_runs, setup.tmax, setup.dt, num_regions, save_percentiles, "ABM", result_dir,
                         draw_func_abm);
    mio::mpm::paper::run(pdmm, num_runs, setup.tmax, setup.dt, num_regions, save_percentiles, "PDMM", result_dir,
                         draw_func_pdmm);
    return 0;
}
