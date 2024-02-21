#include "hybrid_paper/library/quad_well.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/transition_rate_estimation.h"
#include "models/mpm/abm.h"
#include "quad_well_setup.h"
#include "memilio/math/eigen.h"
#include <cstddef>
#include <numeric>
#include <vector>

int main(int argc, char** argv)
{
    using Status = mio::mpm::paper::InfectionState;
    using Model  = mio::mpm::ABM<QuadWellModel<Status>>;

    std::vector<double> pops{2000, 2000, 2000, 2000};
    size_t num_agents = std::accumulate(pops.begin(), pops.end(), 0.0);
    QuadWellSetup<Model::Agent> setup(num_agents);
    setup.adoption_rates.clear();
    Model model = setup.create_abm<Model>();

    auto rates = mio::mpm::paper::calculate_transition_rates(model, 10, setup.tmax, 4, pops);

    return 0;
}
