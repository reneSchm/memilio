#include "hybrid_paper/library/ensemble_run.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/initialization.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "quad_well_setup.h"

int main()
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status = mio::mpm::paper::InfectionState;
    using ABM    = mio::mpm::ABM<QuadWellModel<Status>>;
    using PDMM   = mio::mpm::PDMModel<4, Status>;

    size_t num_runs = 300;
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
    mio::mpm::paper::run(abm, num_runs, setup.tmax, setup.dt, 4, true, "2.5_ABM",
                         mio::base_dir() + "cpp/outputs/QuadWell/Scenario2/", draw_func_abm);
    mio::mpm::paper::run(pdmm, num_runs, setup.tmax, setup.dt, 4, true, "2.5_PDMM",
                         mio::base_dir() + "cpp/outputs/QuadWell/Scenario2/", draw_func_pdmm);
    return 0;
}