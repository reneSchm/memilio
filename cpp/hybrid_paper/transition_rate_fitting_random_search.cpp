#include "memilio/data/analyze_result.h"
#include "models/mpm/potentials/potential_germany.h"
#include "models/mpm/potentials/map_reader.h"
#include "models/mpm/abm.h"
#include "memilio/io/mobility_io.h"
#include "hybrid_paper/weighted_gradient.h"
#include "hybrid_paper/initialization.h"
#include "infection_state.h"

#include <set>

#include <random>
#include <filesystem>
#include <iostream>

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()
#define PRECISION 17

// #include <sys/wait.h>
// #include <unistd.h>

#define restart_timer(timer, description)                                                                              \
    {                                                                                                                  \
        TIME_TYPE new_time = TIME_NOW;                                                                                 \
        std::cout << "\r" << description << " :: " << PRINTABLE_TIME(new_time - timer) << std::endl << std::flush;     \
        timer = new_time;                                                                                              \
    }

struct FittingFunctionSetup {
    using Model =
        mio::mpm::ABM<GradientGermany<mio::mpm::paper::InfectionState>>; //mio::mpm::ABM<GradientGermany<States>>;
    using Status = Model::Status;

    const WeightedGradient::GradientMatrix& gradient; // reference to a WeightedGradient.gradient
    const Eigen::MatrixXi metaregions;
    Eigen::MatrixXd reference_commuters; // number of commuters between regions
    double t_max;
    std::vector<Model::Agent> agents;

    const std::set<std::pair<int, int>> border_pairs;

    // <index>: <County Name> <County ID>
    // 0:   Fürstenfeldbruck    9179
    // 1:   Dachau              9174
    // 2:   Starnberg           9188
    // 3:   München LH          9162
    // 4:   München             9184
    // 5:   Freising            9178
    // 6:   Erding              9177
    // 7:   Ebersberg           9175
    // county_ids are the indices of the counties above in the dict "County" from
    // pycode/memilio-epidata/memilio/epidata/defaultDict.py
    const std::vector<int> county_ids = {233, 228, 242, 238, 223, 232, 231, 229};
    // county population data
    const std::vector<double> reference_populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    // total population in all considered counties
    const double reference_population =
        std::accumulate(reference_populations.begin(), reference_populations.end(), 0.0);

    // uses a weighted potential, loads metaregions and commuter data, prepares agents and reference values.
    // needed for the model setup and error calculation in single_run_mobility_error
    explicit FittingFunctionSetup(WeightedGradient& wg, const std::string& metaregions_pgm_file,
                                  const std::string& mobility_data_directory, const double t_max,
                                  const std::string& agent_init_file)
        : gradient(wg.gradient)
        , metaregions([&metaregions_pgm_file]() {
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
        }())
        , reference_commuters([&mobility_data_directory]() {
            // try and load mobility data
            auto res = get_transition_matrices(mobility_data_directory);
            if (res) { // return the commuter matrix
                return res.value();
            }
            else { /// write error and abort
                mio::log(mio::LogLevel::critical, res.error().formatted_message());
                exit(res.error().code().value());
            }
        }())
        , t_max(t_max)
        , agents() // filled below
        , border_pairs(wg.get_keys())
    {
        read_initialization(agent_init_file, agents);
    }
};

bool commuting_condition(double a, double b)
{
    std::stringstream ss("");
    if (a / b < 1) {
        double proc = 1 - a / b;
        if (proc <= 0.25) {
            return true; // green
        }
    }
    else {
        double proc = 1 - b / a;
        if (proc <= 0.25) {
            return true; // green
        }
    }
    return false;
}

int main()
{

    std::cout << "Current path is " << fs::current_path() << '\n';

    TIME_TYPE timer = TIME_NOW;

    std::string path = "../../";

    WeightedGradient wg(path + "potentially_germany_grad.json", path + "boundary_ids.pgm");
    restart_timer(timer, "# Time for creating weighted potential");

    std::string agent_file = "/group/HPC/Gruppen/PSS/Modelle/Hybrid Models/Papers, Theses, "
                             "Posters/2023_Paper_Spatio-temporal_hybrid/initializations/initSusceptible9375.json";

    FittingFunctionSetup ffs(wg, path + "metagermany.pgm", path + "data/mobility/", 100, agent_file);
    restart_timer(timer, "# Time for fitting function setup");

    auto dist_weights = mio::UniformDistribution<double>::get_instance();
    auto dist_sigmas  = mio::UniformDistribution<double>::get_instance();
    auto dist_slope   = mio::UniformDistribution<double>::get_instance();
    double max_w      = 1000;
    double max_sigma  = 50;

    std::vector<double> weights(ffs.border_pairs.size());
    std::vector<double> sigmas(ffs.metaregions.maxCoeff());
    double slope;

    double flows[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    bool continue_condition      = true;
    double highest_award         = 0;
    const Eigen::Vector2d centre = {wg.gradient.rows() / 2.0, wg.gradient.cols() / 2.0}; // centre of the wg.gradient

    while (continue_condition) {
        continue_condition   = false;
        double current_award = 0;
        //draw weights and sigmas
        for (size_t i = 0; i < weights.size(); ++i) {
            weights[i] = dist_weights(0.0, max_w);
        }
        for (size_t i = 0; i < sigmas.size(); ++i) {
            sigmas[i] = dist_sigmas(0.0, max_sigma);
        }

        slope = dist_slope(0.0, 10.0);

        for (Eigen::Index i = 0; i < wg.gradient.rows(); i++) {
            for (Eigen::Index j = 0; j < wg.gradient.cols(); j++) {
                if (wg.base_gradient(i, j) == Eigen::Vector2d{0, 0}) {
                    auto direction    = (Eigen::Vector2d{i, j} - centre).normalized();
                    wg.gradient(i, j) = slope * direction;
                }
            }
        }

        //apply weights
        wg.apply_weights(weights);

        //make abm run
        using Model = FittingFunctionSetup::Model;

        Model model(ffs.agents, {}, ffs.gradient, ffs.metaregions, {}, sigmas);
        mio::Simulation<Model> sim(model, 0, 0.1);
        sim.advance(ffs.t_max);
        auto& m = sim.get_model();

        //check conditions
        for (int i = 0; i < sigmas.size(); i++) {
            for (int j = 0; j < sigmas.size(); j++) {
                if (i == j) {
                    continue;
                }
                auto commuting_cond = commuting_condition(
                    m.number_transitions({Model::Status::S, mio::mpm::Region(i), mio::mpm::Region(j)}) /
                        m.populations.size(),
                    ffs.reference_commuters(ffs.county_ids[i], ffs.county_ids[j]) * ffs.t_max /
                        (2 * ffs.reference_population));
                if (ffs.reference_commuters(ffs.county_ids[i], ffs.county_ids[j]) * ffs.t_max /
                        (2 * ffs.reference_population) >
                    0.5) {
                    if ((i == 0 && j == 4) || (i == 4 && j == 0)) {
                        continue;
                    }
                    if (!commuting_cond) {
                        continue_condition = true;
                    }
                    else {
                        current_award += 1;
                    }
                }
                else if (commuting_cond) {
                    current_award += 0.5;
                }

                flows[i] -=
                    sim.get_model().number_transitions({Model::Status::S, mio::mpm::Region(i), mio::mpm::Region(j)});
                flows[i] +=
                    sim.get_model().number_transitions({Model::Status::S, mio::mpm::Region(j), mio::mpm::Region(i)});
            }
            if (std::abs(flows[i]) > 50) {
                continue_condition = true;
            }
            else {
                current_award += 2;
            }
        }
        if (current_award >= highest_award) {
            highest_award = current_award;
            std::cout << "Award: " << current_award << "\n";
            std::cout << " -w [";
            for (int i = 0; i < weights.size(); i++)
                std::cout << weights[i] << ((i != weights.size() - 1) ? ", " : "");
            std::cout << "] -s [";
            for (int i = 0; i < sigmas.size(); i++)
                std::cout << sigmas[i] << ((i != sigmas.size() - 1) ? ", " : "");
            std::cout << "] -sl " << slope << "\n" << std::flush;
        }
    }

    restart_timer(timer, "# Time for random search");
    std::cout << "Condition fullfilled!\n";
    std::cout << " -w [";
    for (int i = 0; i < weights.size(); i++)
        std::cout << weights[i] << ((i != weights.size() - 1) ? ", " : "");
    std::cout << "] -s [";
    for (int i = 0; i < sigmas.size(); i++)
        std::cout << sigmas[i] << ((i != sigmas.size() - 1) ? ", " : "");
    std::cout << "] -sl " << slope << "\n" << std::flush;

    return 0;
}
