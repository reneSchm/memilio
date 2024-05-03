#include "hybrid_paper/library/quad_well.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/transition_rate_estimation.h"
#include "memilio/compartments/simulation.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "quad_well_setup.h"
#include "memilio/math/eigen.h"
#include <cstddef>
#include <numeric>
#include <vector>

int main(int argc, char** argv)
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status = mio::mpm::paper::InfectionState;
    using Model  = mio::mpm::ABM<QuadWellModel<Status>>;
    using PDMM   = mio::mpm::PDMModel<4, Status>;

    std::vector<double> pops{2000, 2000, 2000, 2000};
    size_t num_agents = std::accumulate(pops.begin(), pops.end(), 0.0);
    QuadWellSetup<Model::Agent> setup(num_agents);
    setup.adoption_rates.clear();
    Model model = setup.create_abm<Model>();
    mio::Simulation<Model> sim(model, 0.0, setup.dt);
    PDMM pdmm = setup.create_pdmm<PDMM>();

    auto rates = mio::mpm::paper::calculate_transition_rates(model, 300, setup.tmax, 4, pops, 1);
    //auto rates_pdmm = mio::mpm::paper::calculate_transition_rates(pdmm, 300, 100, 4, pops, 1);

    for (size_t r = 0; r < rates.size(); ++r) {
        auto rate = rates[r];
        //auto rate_pdmm = rates_pdmm[r];
        std::cout << rate.from << "->" << rate.to << ": "
                  << " Status: " << static_cast<size_t>(rate.status) << ": " << rate.factor
                  << std::endl; //<< "/" << rate_pdmm.factor << std::endl;
    }

    return 0;
}
