#include "memilio/compartments/simulation.h"
#include "memilio/io/cli.h"
#include "memilio/io/mobility_io.h"
#include "mpm/region.h"
#include "mpm/utility.h"
#include "weighted_gradient.h"
#include "infection_state.h"
#include "models/mpm/abm.h"
#include "models/mpm/potentials/potential_germany.h"
#include "hybrid_paper/initialization.h"
#include <cstdio>
#include <numeric>
#include <sstream>
#include <vector>

mio::IOResult<Eigen::MatrixXd> get_transition_matrices(std::string data_dir)
{
    BOOST_OUTCOME_TRY(matrix_commuter, mio::read_mobility_plain(data_dir + "/commuter_migration_scaled.txt"));
    BOOST_OUTCOME_TRY(matrix_twitter, mio::read_mobility_plain(data_dir + "/twitter_scaled_1252.txt"));
    Eigen::MatrixXd travel_to_matrix = matrix_commuter + matrix_twitter;
    Eigen::MatrixXd transitions_per_day(travel_to_matrix.rows(), travel_to_matrix.cols());
    for (int from = 0; from < travel_to_matrix.rows(); ++from) {
        for (int to = 0; to < travel_to_matrix.cols(); ++to) {
            transitions_per_day(from, to) = travel_to_matrix(from, to) + travel_to_matrix(to, from);
        }
    }
    return mio::success(transitions_per_day);
}

struct Weights {
    using Type = std::vector<double>;
    const static std::string name()
    {
        return "Weights";
    }
    const static std::string alias()
    {
        return "w";
    }
};
struct Sigmas {
    using Type = std::vector<double>;
    const static std::string name()
    {
        return "Sigmas";
    }
    const static std::string alias()
    {
        return "s";
    }
};

std::string colorize(double a, double b)
{
    std::stringstream ss("");
    if (a / b < 1) {
        double proc = 1 - a / b;
        if (proc <= 0.25) {
            ss << "\033[32m"; // green
        }
        else if (proc <= 0.50) {
            ss << "\033[33m"; // yellow
        }
        else {
            ss << "\033[31m"; // red
        }
    }
    else {
        double proc = 1 - b / a;
        if (proc <= 0.25) {
            ss << "\033[42m"; // green
        }
        else if (proc <= 0.50) {
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
    using Model = mio::mpm::ABM<GradientGermany<mio::mpm::paper::InfectionState>>;

    auto cli_result = mio::command_line_interface<Weights, Sigmas>(argv[0], argc, argv);
    if (!cli_result) {
        std::cerr << cli_result.error().formatted_message() << "\n";
        return cli_result.error().code().value();
    }
    auto weights = cli_result.value().get<Weights>();
    auto sigmas  = cli_result.value().get<Sigmas>();

    WeightedGradient wg("../../../potentially_germany_grad.json", "../../../boundary_ids.pgm");

    std::string agent_file = "/group/HPC/Gruppen/PSS/Modelle/Hybrid Models/Papers, Theses, "
                             "Posters/2023_Paper_Spatio-temporal_hybrid/initializations/initSusceptible9375.json";
    std::vector<Model::Agent> agents;
    read_initialization(agent_file, agents);

    wg.apply_weights(weights);

    auto metaregions_pgm_file = "../../../metagermany.pgm";
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

    Model m(agents, {}, wg.gradient, metaregions, {}, sigmas);

    double tmax = 100;

    mio::Simulation<Model> sim(m, 0, 0.1);
    sim.advance(tmax);
    auto results = sim.get_result();

    const std::vector<int> county_ids   = {233, 228, 242, 238, 223, 232, 231, 229};
    Eigen::MatrixXd reference_commuters = get_transition_matrices("../../../data/mobility/").value();
    auto ref_pops = std::vector<double>{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    auto ref_pop  = std::accumulate(ref_pops.begin(), ref_pops.end(), 0.0);

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
                          << (reference_commuters(county_ids[i], county_ids[j]) * tmax / (2 * ref_pop) > 0.5 ? "#"
                                                                                                             : " ")
                          << colorize(sim.get_model().number_transitions(
                                          {Model::Status::S, mio::mpm::Region(i), mio::mpm::Region(j)}) /
                                          sim.get_model().populations.size(),
                                      reference_commuters(county_ids[i], county_ids[j]) * tmax / (2 * ref_pop))
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

    auto file = fopen("/home/schm_r6/Documents/masterwork/output.txt", "w");
    mio::mpm::print_to_file(file, results, {});
    fclose(file);

    return 0;
}