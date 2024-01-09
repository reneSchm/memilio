#include "hybrid_paper/library/potentials/potential_germany.h"
#include "hybrid_paper/library/map_reader.h"

#include <fstream>

enum class InfectionState
{
    S,
    I,
    R,
    Count
};

int main()
{
    using namespace mio::mpm;
    using Model     = ABM<PotentialGermany<InfectionState>>;
    using Status    = Model::Status;
    size_t n_agents = 1;

    std::vector<Model::Agent> agents(n_agents);

    std::vector<double> pop_dist{0.95, 0.05, 0.0};
    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    auto& sta_rng = mio::DiscreteDistribution<int>::get_instance();
    for (auto& a : agents) {
        // a.position = {pos_rng(-2.0, 2.0), pos_rng(-2.0, 2.0)};
        // if (a.position[0] < 0) {
        //     a.status = static_cast<Status>(sta_rng(pop_dist));
        // }
        // else {
        //     a.status = Status::S;
        // }
        a.status   = Status::S;
        a.position = {800, 800};
    }
    // avoid edge cases caused by random starting positions
    // for (auto& agent : agents) {
    //     for (int i = 0; i < 5; i++) {
    //         Model::move(0, 0.1, agent);
    //     }
    // }

    std::vector<AdoptionRate<Status>> adoption_rates;
    for (int i = 0; i < 16; i++) {
        adoption_rates.push_back({Status::S, Status::I, mio::mpm::Region(i), 0.3, {Status::I}, {1}});
        adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), 0.1});
    }

    Eigen::MatrixXd potential;
    Eigen::MatrixXi metaregions;

    std::cerr << "Setup: Read potential.\n" << std::flush;
    {
        const auto fname = "../../../potentially_germany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            return 1;
        }
        else {
            potential = 8 * read_pgm(ifile);
            ifile.close();
        }
    }
    std::cerr << "Setup: Read metaregions.\n" << std::flush;
    {
        const auto fname = "../../../metagermany.pgm";
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

    Model model(agents, adoption_rates, potential, metaregions);

    std::cerr << "Starting simulation.\n" << std::flush;

    auto result = mio::simulate(0, 100, 0.05, model);

    // mio::mpm::print_to_terminal(result, {"S", "I", "R", "S", "I", "R"});

    return 0;
}
