#include "memilio/data/analyze_result.h"
#include "models/mpm/potentials/potential_germany.h"
#include "models/mpm/potentials/map_reader.h"
#include "models/mpm/abm.h"
#include "memilio/io/mobility_io.h"
#include "hybrid_paper/weighted_gradient.h"

#include <dlib/global_optimization.h>
#include <omp.h>

#include <set>

#include <filesystem>
#include <iostream>

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()
#define PRECISION 17

// #include <sys/wait.h>
// #include <unistd.h>

// the minimum needed "Infection States", since we do not consider adoption processes here
enum class States
{
    Default,
    Count
};

// load mobility data between all german counties
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

struct FittingFunctionSetup {
    using Model  = mio::mpm::ABM<GradientGermany<States>>;
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
                                  const std::string& mobility_data_directory, const double t_max)
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
        // set one agent per every four pixels
        // the concentration of agents does not appear to have a strong influence on the error of single_run_mobility_error below
        //std::vector<double> subpopulations(reference_populations.begin(), reference_populations.end());
        // while (std::accumulate(subpopulations.begin(), subpopulations.end(), 0.0) > 0) {
        for (Eigen::Index i = 0; i < metaregions.rows(); i += 2) {
            for (Eigen::Index j = 0; j < metaregions.cols(); j += 2) {
                // auto& pop = subpopulations[metaregions(i, j) - 1];
                if (metaregions(i, j) != 0) {
                    // if (metaregions(i, j) != 0 && pop > 0) {
                    agents.push_back({{i, j}, Model::Status::Default, metaregions(i, j) - 1});
                    // --pop;
                }
            }
        }
        // }
    }
};

// creates a model, runs it, and calculates the l2 error for transition rates
double single_run_mobility_error(const FittingFunctionSetup& ffs, const std::vector<double>& sigma)
{
    TIME_TYPE pre_run = TIME_NOW;
    using Model       = FittingFunctionSetup::Model;

    // create model
    Model model(ffs.agents, {}, ffs.gradient, ffs.metaregions, {}, sigma);

    // run simulation
    mio::Simulation<Model> sim(model, 0, 0.05);
    sim.advance(ffs.t_max);
    auto result = sim.get_result();

    // shorthand for model
    auto& m = sim.get_model();

    // calculate and return error
    double l_2 = 0;
    // double l_inf = 0;
    for (auto& key : ffs.border_pairs) {
        auto from      = key.first;
        auto to        = key.second;
        const auto val = m.number_transitions({Model::Status::Default, mio::mpm::Region(from), mio::mpm::Region(to)}) /
                         (m.populations.size() * (sim.get_result().get_last_time() - sim.get_result().get_time(0)));
        const auto ref =
            ffs.reference_commuters(ffs.county_ids[from], ffs.county_ids[to]) / (2 * ffs.reference_population);
        const auto err = std::abs(val - ref);
        l_2 += err * err;
        // l_inf = std::max(l_inf, err);
        // std::cout << from << "->" << to << ":  value=";
        // set_ostream_format(std::cout) << val << "  error=";
        // set_ostream_format(std::cout) << err << "  rel_error=";
        // set_ostream_format(std::cout) << ((ref > 0) ? err / ref : 0) << "\n";
    }
    TIME_TYPE post_run = TIME_NOW;
    fprintf(stdout, "# Time for one run: %.*g\n", PRECISION, PRINTABLE_TIME(post_run - pre_run));
    // std::cout << "l2 err=" << std::sqrt(l_2) << "\n";
    // std::cout << "linf err=" << l_inf << "\n";
    return std::sqrt(l_2);
}

double average_run_mobility_error(const FittingFunctionSetup& ffs, const std::vector<double>& sigma, int num_runs)
{
    std::vector<double> errors(num_runs);
#pragma omp parallel for
    for (int run = 0; run < num_runs; ++run) {
        errors[run] = single_run_mobility_error(ffs, sigma);
    }
    return std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
}

int main()
{
    int num_runs = (omp_get_max_threads() > 1) ? omp_get_max_threads() : 3;
    std::cout << "num_runs = " << num_runs << "\n";

    std::cout << "Current path is " << fs::current_path() << '\n';
    TIME_TYPE pre_potential = TIME_NOW;
    WeightedGradient wg("../../potentially_germany_grad.json", "../../boundary_ids.pgm");
    TIME_TYPE post_potential = TIME_NOW;
    fprintf(stdout, "# Time for creating weighted potential: %.*g\n", PRECISION,
            PRINTABLE_TIME(post_potential - pre_potential));
    TIME_TYPE pre_fitting_function_setup = TIME_NOW;
    FittingFunctionSetup ffs(wg, "../../metagermany.pgm", "../../data/mobility/", 2);
    TIME_TYPE post_fitting_function_setup = TIME_NOW;
    fprintf(stdout, "# Time for fitting function setup: %.*g\n", PRECISION,
            PRINTABLE_TIME(post_fitting_function_setup - pre_fitting_function_setup));

    TIME_TYPE pre_min_function = TIME_NOW;
    auto result                = dlib::find_min_global(
        [&](auto&& w1, auto&& w2, auto&& w3, auto&& w4, auto&& w5, auto&& w6, auto&& w7, auto&& w8, auto&& w9,
            auto&& w10, auto&& w11, auto&& w12, auto&& w13, auto&& w14, auto&& sigma1, auto&& sigma2, auto&& sigma3,
            auto&& sigma4, auto&& sigma5, auto&& sigma6, auto&& sigma7, auto&& sigma8) {
            // let dlib set the weights for the potential
            wg.apply_weights({w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14});
            // calculate the transition rate error
            return average_run_mobility_error(ffs, {sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8},
                                                             num_runs);
        },
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, // lower bounds
        {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15, 15, 15, 15}, // upper bounds
        std::chrono::hours(24) // run this long
    );
    TIME_TYPE post_min_function = TIME_NOW;
    fprintf(stdout, "# Time for minimizing: %.*g\n", PRECISION, PRINTABLE_TIME(post_min_function - pre_min_function));

    std::cout << "Minimizer borders:\n";
    for (size_t border = 0; border < 14; ++border) {
        std::cout << result.x(border) << "\n";
    }
    std::cout << "Minimizer sigma: \n";
    for (size_t s = 14; s < result.x.size(); ++s) {
        std::cout << result.x(s) << "\n";
    }
    std::cout << "Minimum error:\n" << result.y << "\n";

    return 0;
}
