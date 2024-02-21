#include "hybrid_paper/library/model_setup.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "hybrid_paper/library/weighted_gradient.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/map_reader.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/potentials/potential_germany.h"
#include "hybrid_paper/library/transition_rate_estimation.h"
#include "mpm/abm.h"
#include "mpm/model.h"

#include "memilio/io/cli.h"

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
    using Model  = mio::mpm::ABM<CommutingPotential<Kedaechtnislos, mio::mpm::paper::InfectionState>>;
    using Status = mio::mpm::paper::InfectionState;

    mio::mpm::paper::ModelSetup<Model::Agent> setup;
    Model model = setup.create_abm<Model>();

    //print_movement<Model>(agents, model, tmax, 0.1);

    auto rates =
        mio::mpm::paper::calculate_transition_rates(model, 10, setup.tmax, setup.region_ids.size(), setup.populations);

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
