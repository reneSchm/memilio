#include "hybrid_paper/library/model_setup.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "hybrid_paper/library/weighted_gradient.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/transition_rate_estimation.h"
#include "memilio/utils/logging.h"
#include "mpm/abm.h"
#include "mpm/model.h"
#include "mpm/pdmm.h"
#include "ode_secir/model.h"

#include <numeric>
#include <vector>

struct Tmax {
    using Type = double;
    const static std::string name()
    {
        return "Tmax";
    }
    const static std::string alias()
    {
        return "tmax";
    }
};

template <class Model>
void print_movement(std::vector<typename Model::Agent> agents, Model& model, const double tmax, double dt)
{
    double t = 0;
    auto sim = mio::Simulation<Model>(model, t, dt);
    while (t < tmax) {
        for (auto& agent : sim.get_model().populations) {
            std::cout << agent.position[0] << " " << agent.position[1] << " ";
            sim.get_model().move(t, dt, agent);
        }
        std::cout << "\n";
        t += dt;
    }
}

std::string colorize(double a, double b)
{
    std::stringstream ss("");
    if (a / b < 1) {
        double proc = 1 - a / b;
        if (proc <= 0.05) {
            ss << "\033[32m"; // green
        }
        else if (proc <= 0.15) {
            ss << "\033[33m"; // yellow
        }
        else {
            ss << "\033[31m"; // red
        }
    }
    else {
        double proc = 1 - b / a;
        if (proc <= 0.05) {
            ss << "\033[42m"; // green
        }
        else if (proc <= 0.15) {
            ss << "\033[43m"; // yellow
        }
        else {
            ss << "\033[41m"; // red
        }
    }
    ss << a << " / " << b << "\033[0m";
    return ss.str();
}

int main(int argc, char** argv)
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status = mio::mpm::paper::InfectionState;
    using ABM    = mio::mpm::ABM<CommutingPotential<Kedaechtnislos, Status>>;
    using PDMM   = mio::mpm::PDMModel<8, Status>;

    using Model = PDMM;

    mio::mpm::paper::ModelSetup<ABM::Agent> setup;
    setup.adoption_rates.clear();
    for (int k = 0; k < 8; k++) {
        setup.pop_dists_scaled[k][0] =
            std::accumulate(setup.pop_dists_scaled[k].begin(), setup.pop_dists_scaled[k].end(), 0.0);
        for (int s = 1; s < (int)Status::Count; s++) {
            setup.pop_dists_scaled[k][s] = 0;
        }
    }
    Model model = setup.create_pdmm<Model>();

    // ({setup.commute_weights, setup.metaregions, {setup.metaregions}, setup.persons_per_agent}, setup.agents,            setup.adoption_rates, setup.wg.gradient, setup.metaregions, {Status::D}, setup.sigmas,setup.contact_radius);

    //print_movement<Model>(agents, model, tmax, 0.1);

    auto rates = mio::mpm::paper::calculate_transition_rates(model, 10, setup.tmax, setup.region_ids.size(),
                                                             setup.populations, setup.persons_per_agent);

    //check
    for (auto rate : rates) {
        //rates
        std::cout << rate.from << "->" << rate.to << ": rel: "
                  << colorize(rate.factor,
                              (setup.commute_weights(static_cast<size_t>(rate.from), static_cast<size_t>(rate.to)) +
                               setup.commute_weights(static_cast<size_t>(rate.to), static_cast<size_t>(rate.from))) /
                                  setup.populations[rate.from.get()])
                  << std::endl;
    }

    return 0;
}
