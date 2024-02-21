#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/quad_well.h"
#include "mpm/abm.h"
#include "mpm/pdmm.h"
#include "memilio/math/eigen.h"

#include <omp.h>

void run_simulation(size_t num_runs, bool save_percentiles, bool save_single_outputs, std::string result_path)
{

    const size_t num_regions = 4;
    using Status             = mio::mpm::paper::InfectionState;
    using Region             = mio::mpm::Region;
    using ABM                = mio::mpm::ABM<QuadWellModel<Status>>;
    using PDMM               = mio::mpm::PDMModel<4, Status>;
}

int main()
{
    return 0;
}