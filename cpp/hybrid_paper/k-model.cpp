#include "abm/time.h"
#include "hybrid_paper/lib/infection_state.h"
#include "hybrid_paper/lib/initialization.h"
#include "hybrid_paper/lib/weighted_gradient.h"
#include "hybrid_paper/lib/metaregion_sampler.h"
#include "hybrid_paper/lib/map_reader.h"
#include "hybrid_paper/lib/potentials/commuting_potential.h"
#include "hybrid_paper/lib/potentials/potential_germany.h"

#include "memilio/config.h"
#include "memilio/math/eigen.h"
#include "memilio/math/floating_point.h"
#include "memilio/utils/parameter_distributions.h"
#include "memilio/utils/random_number_generator.h"
#include "memilio/utils/span.h"
#include "memilio/utils/time_series.h"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

std::string colorize(double a, double b)
{
    std::stringstream ss("");
    if (a / b < 1) {
        double proc = 1 - a / b;
        if (proc <= 0.1) {
            ss << "\033[32m"; // green
        }
        else if (proc <= 0.25) {
            ss << "\033[33m"; // yellow
        }
        else {
            ss << "\033[31m"; // red
        }
    }
    else {
        double proc = 1 - b / a;
        if (proc <= 0.1) {
            ss << "\033[42m"; // green
        }
        else if (proc <= 0.25) {
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
    using KProvider = StochastiK;
    using Model     = mio::mpm::ABM<CommutingPotential<KProvider, mio::mpm::paper::InfectionState>>;

    auto weights = std::vector<double>(14, 500);
    auto sigmas  = std::vector<double>(8, 10);
    // double slope = 0; unused

    WeightedGradient wg(mio::base_dir() + "potentially_germany_grad.json", mio::base_dir() + "boundary_ids.pgm");

    std::string agent_file = "/group/HPC/Gruppen/PSS/Modelle/Hybrid Models/Papers, Theses, "
                             "Posters/2023_Paper_Spatio-temporal_hybrid/initializations/initSusceptible9375.json";
    std::vector<Model::Agent> agents;
    read_initialization(agent_file, agents);

    wg.apply_weights(weights);

    auto metaregions_pgm_file = mio::base_dir() + "metagermany.pgm";
    auto metaregions          = [&metaregions_pgm_file]() {
        std::ifstream ifile(metaregions_pgm_file);
        if (!ifile.is_open()) { // write error and abort
            mio::log(mio::LogLevel::critical, "Could not open metaregion file {}", metaregions_pgm_file);
            exit(1);
        }
        else { // read pgm from file and return matrix
            auto tmp = mio::mpm::read_pgm_raw(ifile).first;
            ifile.close();
            return tmp;
        }
    }();

    const std::vector<int> county_ids   = {233, 228, 242, 223, 238, 232, 231, 229};
    Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();
    auto ref_pops = std::vector<double>{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    auto ref_pop  = std::accumulate(ref_pops.begin(), ref_pops.end(), 0.0);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(8, 8);
    commute_weights.setZero();
    for (int i = 0; i < 8; i++) {
        commute_weights(i, i) = 1;
    }
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i != j) {
                commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
            }
        }
        commute_weights(i, i) = ref_pops[i] - commute_weights.row(i).sum();
    }

    // TODO: adjust weights to daily commuter rates
    StochastiK k_provider(commute_weights, metaregions, {metaregions});

    double tmax = 100;

    Model m(k_provider, agents, {}, wg.gradient, metaregions, {}, sigmas);

    mio::Simulation<Model> sim(m, 0, 0.1);
    sim.advance(tmax);
    auto results = sim.get_result();

    double flows[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    std::cout << "#############\n";

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i != j &&
                sim.get_model().number_transitions({Model::Status::S, mio::mpm::Region(i), mio::mpm::Region(j)})) {
                flows[i] -=
                    sim.get_model().number_transitions({Model::Status::S, mio::mpm::Region(i), mio::mpm::Region(j)});
                flows[i] +=
                    sim.get_model().number_transitions({Model::Status::S, mio::mpm::Region(j), mio::mpm::Region(i)});

                std::cout << i << "->" << j << " : "
                          << (reference_commuters(county_ids[i], county_ids[j]) * tmax / ref_pop > 0.5 ? "#" : " ")
                          << colorize(sim.get_model().number_transitions(
                                          {Model::Status::S, mio::mpm::Region(i), mio::mpm::Region(j)}), //
                                      (reference_commuters(county_ids[i], county_ids[j]) +
                                       reference_commuters(county_ids[j], county_ids[i])) *
                                          tmax * sim.get_model().populations.size() / ref_pop) //
                          << "\n";
            }
        }
    }

    int count = 0;
    for (auto a : sim.get_model().populations) {
        if (metaregions(a.position[0], a.position[1]) == 0) {
            count++;
        }
    }

    std::cout << "#############\nleft out : " << count << "\n";

    std::cout << "flows:\n";
    for (int i = 0; i < 8; i++) {
        std::cout << i << ": " << flows[i] << "\n";
    }

    std::cout << "\n" << sim.get_model().number_transitions()[0] << "\n";

    bool print_output = false;
    if (print_output) {
        auto file = fopen((mio::base_dir() + "output.txt").c_str(), "w");
        mio::mpm::print_to_file(file, results, {});
        fclose(file);
    }

    return 0;
}