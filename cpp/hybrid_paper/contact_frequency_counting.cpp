#include "initialization.h"
#include "mpm/potentials/commuting_potential.h"
#include "hybrid_paper/infection_state.h"
#include "mpm/abm.h"
#include "weighted_gradient.h"
#include <iostream>

int main(int argc, char** argv)
{
    using Status       = mio::mpm::paper::InfectionState;
    using Model        = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    size_t num_regions = 8;

    const double tmax = 30;
    std::vector<double> sigmas(10, num_regions);
    std::vector<double> weights(14, 500);
    std::string agent_init_file = "/group/HPC/Gruppen/PSS/Modelle/Hybrid Models/Papers, Theses, "
                                  "Posters/2023_Paper_Spatio-temporal_hybrid/initializations/initSusceptible4016.json";
    const double contact_radius = 5;

    WeightedGradient wg(mio::base_dir() + "potentially_germany_grad.json", mio::base_dir() + "boundary_ids.pgm");
    wg.apply_weights(weights);

    Eigen::MatrixXi metaregions;
    {
        const auto fname = mio::base_dir() + "metagermany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            return 1;
        }
        else {
            metaregions = mio::mpm::read_pgm_raw(ifile).first;
            ifile.close();
        }
    }

    const std::vector<int> county_ids   = {233, 228, 242, 223, 238, 232, 231, 229};
    Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();
    auto ref_pops = std::vector<double>{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    auto ref_pop  = std::accumulate(ref_pops.begin(), ref_pops.end(), 0.0);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(num_regions, num_regions);
    commute_weights.setZero();
    for (int i = 0; i < num_regions; i++) {
        for (int j = 0; j < num_regions; j++) {
            if (i != j) {
                commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
            }
        }
        commute_weights(i, i) = ref_pops[i] - commute_weights.row(i).sum();
    }

    StochastiK k_provider(commute_weights, metaregions, {metaregions});
    //std::cerr << k_provider.metaregion_commute_weights << "\n";

    std::vector<Model::Agent> agents;
    read_initialization(agent_init_file, agents);
    size_t num_agents = agents.size();

    Model model(k_provider, agents, {}, wg.gradient, metaregions, {Status::D}, sigmas, contact_radius);

    double accumulated_contact_rates = 0.0;
    int num_contacts;
    int num_countings = 0;

    double t  = 0;
    double dt = 0.1;
    auto sim  = mio::Simulation<Model>(model, t, dt);
    while (t < tmax) {
        for (auto& agent : sim.get_model().populations) {
            num_contacts = 0;
            for (auto& contact : sim.get_model().populations) {
                if (sim.get_model().is_contact(agent, contact)) {
                    num_contacts++;
                }
            }
            if (num_contacts > 0) {
                accumulated_contact_rates += (num_contacts / sim.get_model().get_total_agents_in_region(agent.region));
                num_countings += 1;
            }
            sim.get_model().move(t, dt, agent);
        }
        t += dt;
    }

    std::cout << "num agents: " << num_agents << " contact radius:" << contact_radius
              << " scaling factor: " << num_countings / accumulated_contact_rates << "\n";
}