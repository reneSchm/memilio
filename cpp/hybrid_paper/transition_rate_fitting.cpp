#include "hybrid_paper/initialization.h"
#include "hybrid_paper/weighted_gradient.h"
#include "models/mpm/potentials/potential_germany.h"
#include "models/mpm/potentials/map_reader.h"
#include "models/mpm/abm.h"
#include "memilio/data/analyze_result.h"
#include "memilio/io/mobility_io.h"

#include <dlib/threads.h>
#include <dlib/global_optimization.h>
#include <omp.h>

#include <set>
#include <filesystem>
#include <iostream>
#include <vector>

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
    explicit FittingFunctionSetup(const WeightedGradient& wg, const std::string& metaregions_pgm_file,
                                  const std::string& mobility_data_directory, const std::string& agent_init_file,
                                  const double t_max)
        : metaregions([&metaregions_pgm_file]() {
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
        // set one agent per every four pixels
        // the concentration of agents does not appear to have a strong influence on the error of single_run_mobility_error below
        //std::vector<double> subpopulations(reference_populations.begin(), reference_populations.end());
        // while (std::accumulate(subpopulations.begin(), subpopulations.end(), 0.0) > 0) {
        // for (Eigen::Index i = 0; i < metaregions.rows(); i += 2) {
        //     for (Eigen::Index j = 0; j < metaregions.cols(); j += 2) {
        //         // auto& pop = subpopulations[metaregions(i, j) - 1];
        //         if (metaregions(i, j) != 0) {
        //             // if (metaregions(i, j) != 0 && pop > 0) {
        //             agents.push_back({{i, j}, Model::Status::Default, metaregions(i, j) - 1});
        //             // --pop;
        //         }
        //     }
        // }
        // }
    }
};

// creates a model, runs it, and calculates the l2 error for transition rates
double single_run_mobility_error(const FittingFunctionSetup& ffs,
                                 Eigen::Ref<const WeightedGradient::GradientMatrix> gradient,
                                 const std::vector<double>& sigma)
{
    TIME_TYPE pre_run = TIME_NOW;
    using Model       = FittingFunctionSetup::Model;

    // create model
    Model model(ffs.agents, {}, gradient, ffs.metaregions, {}, sigma);

    // run simulation
    mio::Simulation<Model> sim(model, 0, 0.1);
    sim.advance(ffs.t_max);

    // shorthand for model
    auto& m = sim.get_model();

    // calculate and return error
    double l_2 = 0;
    const double error_weight =
        1; // keeps norm-equivalence to l2, but lets pop_change have a weaker influence on total error
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
    }
#ifdef USE_POPULATION_CHANGE_ERROR // see memilio/config.h
    // also consider population change (goal : keep metapopulation sizes constant)
    auto pop_t0   = sim.get_result().get_value(0);
    auto pop_tmax = sim.get_result().get_last_value();
    auto pop_change =
        ((pop_t0 - pop_tmax).array() / pop_t0.array()).square().sum(); // .square().sum() is same as .norm()^2
    l_2 += error_weight * pop_change;
#endif
    return std::sqrt(l_2);
}

double average_run_mobility_error(const FittingFunctionSetup& ffs, const WeightedGradient& wg,
                                  const std::vector<double>& sigma, int num_runs)
{
    std::vector<double> errors(num_runs);
#pragma omp parallel for num_threads(num_runs)
    for (int run = 0; run < num_runs; ++run) {
        errors[run] = single_run_mobility_error(ffs, wg.gradient, sigma);
    }
    return std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
}

Eigen::Ref<WeightedGradient::GradientMatrix> thread_specific_gradient(Eigen::Index rows, Eigen::Index cols)
{
    static thread_local WeightedGradient::GradientMatrix grad =
        WeightedGradient::GradientMatrix::Constant(rows, cols, WeightedGradient::GradientMatrix::Scalar::Zero());
    return grad;
}

int main()
{
    // NOTE: Use DLIB_NUM_THREADS to set a maximum number of threads
    const std::string path        = "";
    const double solver_epsilon   = 0.1;
    const double tmax             = 20;
    const auto total_fitting_time = std::chrono::hours(140);
    const double weight_min = 0, weight_max = 1000;
    const double sigma_min = 0, sigma_max = 50;
    const double slope_min = 0, slope_max = 10;

    double max_error       = 0;
    double acc_error       = 0;
    size_t num_errors      = 0;
    auto total_num_threads = dlib::default_thread_pool().num_threads_in_pool();
    dlib::mutex print_mutex; // used to prevent parallel prints from bleeding into each other
    // see also http://dlib.net/api.html#threads or http://dlib.net/threads_ex.cpp.html

#ifdef USE_MICROSTEPPING
    std::cout << "Microstepping:           Enabled\n";
#else
    std::cout << "Microstepping:           Disabled\n";
#endif
#ifdef USE_POPULATION_CHANGE_ERROR
    std::cout << "Population change error: Enabled\n";
#else
    std::cout << "Population change error: Disabled\n";
#endif
    std::cout << "Current path is          " << fs::current_path() << "\n"
              << "ABM tmax:                " << tmax << "\n"
              << "Total threads:           " << total_num_threads << "\n"
              << "Total fitting time:      " << PRINTABLE_TIME(total_fitting_time) << "\n"
              << "Solver epsilon:          " << solver_epsilon << "\n"
              << "\n";

    TIME_TYPE timer = TIME_NOW;

    const WeightedGradient wg(path + "potentially_germany_grad.json", path + "boundary_ids.pgm");
    restart_timer(timer, "# Time for creating weighted potential");

    const FittingFunctionSetup ffs(wg, path + "metagermany.pgm", path + "data/mobility/",
                                   "/group_KP/HPC/Gruppen/PSS/Modelle/Hybrid Models/Papers, Theses, "
                                   "Posters/2023_Paper_Spatio-temporal_hybrid/initializations/initSusceptible9375.json",
                                   tmax);
    restart_timer(timer, "# Time for fitting function setup");

    const Eigen::Vector2d centre = {wg.gradient.rows() / 2.0, wg.gradient.cols() / 2.0}; // centre of the wg.gradient

    auto result = dlib::find_min_global(
        dlib::default_thread_pool(), // use DLIB_NUM_THREADS (or nproc) threads
        [&](auto&& w1, auto&& w2, auto&& w3, auto&& w4, auto&& w5, auto&& w6, auto&& w7, auto&& w8, auto&& w9,
            auto&& w10, auto&& w11, auto&& w12, auto&& w13, auto&& w14, auto&& sigma1, auto&& sigma2, auto&& sigma3,
            auto&& sigma4, auto&& sigma5, auto&& sigma6, auto&& sigma7, auto&& sigma8, auto&& slope) {
            auto gradient = thread_specific_gradient(wg.gradient.rows(), wg.gradient.cols());
            // set the weights for the potential
            wg.apply_weights({w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14}, gradient);
            // set an interior slope
            for (Eigen::Index i = 0; i < gradient.rows(); i++) {
                for (Eigen::Index j = 0; j < gradient.cols(); j++) {
                    if (wg.base_gradient(i, j) == Eigen::Vector2d{0, 0}) {
                        auto direction = (Eigen::Vector2d{i, j} - centre).normalized();
                        gradient(i, j) = slope * direction;
                    }
                }
            }
            // calculate the transition rate error
            auto err = single_run_mobility_error(ffs, gradient,
                                                 {sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7, sigma8});
            // print out the used config and calculated error to std::err
            dlib::auto_mutex lock_printing_until_end_of_scope(print_mutex); // locks on ctor, unlocks on destructor
            std::cerr << " -w [" << w1 << ", " << w2 << ", " << w3 << ", " << w4 << ", " << w5 << ", " << w6 << ", "
                      << w7 << ", " << w8 << ", " << w9 << ", " << w10 << ", " << w11 << ", " << w12 << ", " << w13
                      << ", " << w14 << "]";
            std::cerr << " -s [" << sigma1 << ", " << sigma2 << ", " << sigma3 << ", " << sigma4 << ", " << sigma5
                      << ", " << sigma6 << ", " << sigma7 << ", " << sigma8 << "]";
            std::cerr << " -sl " << slope << "\n";
            std::cerr << "E: " << err << "\n\n";
            // gather some statistics
            max_error = std::max(max_error, err);
            acc_error += err;
            num_errors++;
            // return the error
            return err;
        },
        {weight_min, weight_min, weight_min, weight_min, weight_min, weight_min, weight_min, weight_min,
         weight_min, weight_min, weight_min, weight_min, weight_min, weight_min, sigma_min,  sigma_min,
         sigma_min,  sigma_min,  sigma_min,  sigma_min,  sigma_min,  sigma_min,  slope_min}, // lower bounds
        {weight_max, weight_max, weight_max, weight_max, weight_max, weight_max, weight_max, weight_max,
         weight_max, weight_max, weight_max, weight_max, weight_max, weight_max, sigma_max,  sigma_max,
         sigma_max,  sigma_max,  sigma_max,  sigma_max,  sigma_max,  sigma_max,  slope_max}, // upper bounds
        total_fitting_time, // run this long
        solver_epsilon // target accuracy for searching local minima, before returning to global search
    );

    restart_timer(timer, "# Time for minimizing");

    std::cout << "Minimizer borders:\n";
    for (size_t border = 0; border < 14; ++border) {
        std::cout << result.x(border) << "\n";
    }
    std::cout << "Minimizer sigma: \n";
    for (size_t s = 14; s < result.x.size(); ++s) {
        std::cout << result.x(s) << "\n";
    }
    std::cout << "Minimizer evaluations:\n" << num_errors << "\n";
    std::cout << "Minimum error:\n" << result.y << "\n";
    std::cout << "Average error:\n" << acc_error / num_errors << "\n";
    std::cout << "Maximum error:\n" << max_error << "\n";

    return 0;
}
