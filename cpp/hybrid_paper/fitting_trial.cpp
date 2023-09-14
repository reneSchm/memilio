#include "memilio/data/analyze_result.h"
#include "models/mpm/potentials/potential_germany.h"
#include "models/mpm/potentials/map_reader.h"
#include "models/mpm/abm.h"

#include <sys/wait.h>
#include <unistd.h>

enum class States
{
    Default,
    Count
};

int main()
{
    using namespace mio::mpm;
    using Model     = ABM<PotentialGermany<States>>;
    using Status    = Model::Status;
    size_t n_agents = 100;

    std::vector<AdoptionRate<Status>> adoption_rates;
    Eigen::MatrixXd potential;
    Eigen::MatrixXi metaregions;

    pid_t child_pid;
    int wait_status;

    switch (child_pid = fork()) {
    case -1:
        std::cerr << "Failed to create fork.\n";
        return 1;
    case 0: // child process goes here
        execl("../../../a.out", "", "-p", "\"../../../potentially_germany.pgm\"", "-b", "\"../../../boundary_ids.pgm\"",
              "-w", "[]", NULL);
        break;
    default: // parent process goes here
        std::cerr << "Waiting for child process...\n";
        pid_t pid = wait(&wait_status);
        if (pid != child_pid) {
            std::cerr << "Child process failed. Exiting.\n";
            return 1;
        }
        std::cerr << "Running parent process\n";
        break;
    }

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

    int metaregion = 4;
    std::vector<double> avg_rel_disp(999, 0);
    // for (int metaregion = 0; metaregion < metaregions.maxCoeff(); metaregion++)
    for (int k = 0; k < 10; k++) {

        std::vector<Model::Agent> agents;
        for (Eigen::Index i = 0; i < metaregions.rows(); i++) {
            for (Eigen::Index j = 0; j < metaregions.cols(); j++) {
                if (metaregions(i, j) == metaregion + 1) {
                    agents.push_back({{i, j}, Status::Default, metaregion});
                }
            }
        }

        // std::ofstream ofile("../../../agents_pre.txt");
        // for (auto& a : agents) {
        //     ofile << a.position[0] << " " << potential.cols() - a.position[1] << "\n";
        // }
        // ofile.close();

        Model model(agents, adoption_rates, potential, metaregions);

        std::cerr << "Starting simulation for region " << metaregion << " ( " << agents.size() << " agents).\n";

        // mio::Simulation<Model> sim(model, 0, 0.05);
        // sim.advance(10);
        // auto result = sim.get_result();
        auto result = mio::simulate(0, 1000, 0.05, model);

        // ofile.open("../../../agents_post.txt");
        // for (auto& a : sim.get_model().populations) {
        //     ofile << a.position[0] << " " << potential.cols() - a.position[1] << "\n";
        // }
        // ofile.close();

        // print_to_terminal(result, {});
        auto daily_result = mio::interpolate_simulation_result(result);
        for (int i = 1; i < daily_result.get_num_time_points(); i++) {
            avg_rel_disp[i - 1] += (daily_result.get_value(i - 1)[metaregion] - daily_result.get_value(i)[metaregion]) /
                                   daily_result.get_value(i - 1)[metaregion];
        }

        // std::cerr << "Total displacement: " << result.get_value(0)[metaregion] - result.get_last_value()[metaregion]
        //           << ".\n";
    }
    std::cerr << "Daily relative displacement: ";
    for (int i = 0; i < avg_rel_disp.size(); i++) {
        std::cerr << avg_rel_disp[i] / 10 << " ";
    }
    std::cerr << "\n";

    return 0;
}