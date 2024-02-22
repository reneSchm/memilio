#include "hybrid_paper/library/ensemble_run.h"
#include "hybrid_paper/library/infection_state.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "quad_well_setup.h"

int main()
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status = mio::mpm::paper::InfectionState;
    using ABM    = mio::mpm::ABM<QuadWellModel<Status>>;
    using PDMM   = mio::mpm::PDMModel<4, Status>;

    size_t num_runs = 100;

    QuadWellSetup<ABM::Agent> setup(2000);
    ABM abm   = setup.create_abm<ABM>();
    PDMM pdmm = setup.create_pdmm<PDMM>();
    mio::mpm::paper::run(abm, num_runs, setup.tmax, setup.dt, 4, true, "ABM");
    mio::mpm::paper::run(pdmm, num_runs, setup.tmax, setup.dt, 4, true, "PDMM");
    return 0;
}