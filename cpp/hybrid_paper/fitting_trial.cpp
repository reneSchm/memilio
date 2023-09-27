#include "memilio/data/analyze_result.h"
#include "models/mpm/potentials/potential_germany.h"
#include "models/mpm/potentials/map_reader.h"
#include "models/mpm/abm.h"
#include "memilio/io/mobility_io.h"

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

// struct to store and reapply weights to a potential
// reads a base potential and boundary ids from pgm files during construction. the potential can be accessed as a
// public member, and weights are applied to this potential using the apply_weights function
struct WeightedPotential {
    const Eigen::MatrixXd base_potential; // unweighted potential
    const Eigen::MatrixXi boundary_ids; // bondaries to apply weights to
    Eigen::MatrixXd potential;

private:
    std::set<std::pair<int, int>> missing_keys;

public:
    const size_t num_weights;

private:
    std::map<std::pair<int, int>, ScalarType> weight_map;

public:
    // first load base potential and boundary ids, then find needed weight keys and set up weight_map
    explicit WeightedPotential(const std::string& potential_pgm_file, const std::string& boundary_ids_pgm_file)
        // this uses a lot of immediately invoked lambdas
        : base_potential([&potential_pgm_file] {
            // load and read pgm file
            std::ifstream ifile(potential_pgm_file);
            if (!ifile.is_open()) { // write error and abort
                mio::log(mio::LogLevel::critical, "Could not open potential file {}", potential_pgm_file);
                exit(1);
            }
            else { // read pgm from file and return matrix
                Eigen::MatrixXd tmp = 8.0 * mio::mpm::read_pgm(ifile);
                ifile.close();
                return tmp;
            }
        }())
        , boundary_ids([&boundary_ids_pgm_file]() {
            // load and read pgm file
            std::ifstream ifile(boundary_ids_pgm_file);
            if (!ifile.is_open()) { // write error and abort
                mio::log(mio::LogLevel::critical, "Could not open boundary ids file {}", boundary_ids_pgm_file);
                exit(1);
            }
            else { // read pgm from file and return matrix
                auto tmp = mio::mpm::read_pgm_raw(ifile).first;
                ifile.close();
                return tmp;
            }
        }())
        , potential(base_potential) // assign base_potential, as we have no weights yet
        , missing_keys() // missing keys are set during initialisation of num_weights below
        , num_weights([this]() {
            // abuse get_weight to create a list of all needed keys
            std::map<std::pair<int, int>, ScalarType> empty_map; // provide no existing keys, so all are missing
            // test all bitkeys boundary_ids
            for (Eigen::Index i = 0; i < boundary_ids.rows(); i++) {
                for (Eigen::Index j = 0; j < boundary_ids.cols(); j++) {
                    if (boundary_ids(i, j) > 0) // skip non-boundary entries
                        get_weight(empty_map, boundary_ids(i, j), missing_keys);
                }
            }
            return missing_keys.size();
        }())
        , weight_map([this]() {
            // add a map entry for each key found during initialisation of num_weights above.
            // the weights are set to a default value, which will be overwritten by apply_weights()
            std::map<std::pair<int, int>, ScalarType> map;
            for (auto key : missing_keys) {
                map[key] = 0;
            }
            return map;
        }())
    {
        assert(base_potential.cols() == boundary_ids.cols());
        assert(base_potential.rows() == boundary_ids.rows());
    }

    void apply_weights(const std::vector<ScalarType> weights)
    {
        assert(base_potential.cols() == potential.cols());
        assert(base_potential.rows() == potential.rows());
        assert(weights.size() == num_weights);
        // assign weights to map values
        {
            size_t i = 0;
            for (auto& weight : weight_map) {
                weight.second = weights[i];
                i++;
            }
        }
        // recompute potential
        for (Eigen::Index i = 0; i < potential.rows(); i++) {
            for (Eigen::Index j = 0; j < potential.cols(); j++) {
                if (boundary_ids(i, j) > 0) // skip non-boundary entries
                    potential(i, j) = base_potential(i, j) * get_weight(weight_map, boundary_ids(i, j), missing_keys);
            }
        }
    }

    const std::set<std::pair<int, int>>& get_keys()
    {
        return missing_keys;
    }

private:
    // Return the (maximum) weight corresponding to (all) bits in bitkey. Accepts bitkeys with 0, 2 or 3 bits set.
    // Weights are requested from the map w as a pair (a, b), where a and b are the positions of the bits set in
    // bitkey, and a < b. If no entry is present, the weight defaults to 0.
    // Missing pairs are added to the set missing_keys.
    static ScalarType get_weight(const std::map<std::pair<int, int>, ScalarType> w, int bitkey,
                                 std::set<std::pair<int, int>>& missing_keys)
    {
        // in a map (i.e. the potential), the bitkey encodes which (meta)regions are adjacent to a given boundary
        // point. if a boundary point is next to (or at least close to) the i-th region (index starting at 1), then
        // the (i-1)-th bit in bitkey is set to 1. we only treat the cases that there are two or three adjacent
        // regions, as other cases do not appear in the maps we consider.

        auto num_bits = mio::mpm::num_bits_set(bitkey);
        if (num_bits == 2) { // border between two regions
            // read positions (key) from bitkey
            std::array<int, 2> key;
            int i = 0; // key index
            // iterate bit positions
            for (int k = 0; k < 8 * sizeof(int); k++) {
                if ((bitkey >> k) & 1) { // check that the k-th bit is set
                    // write down keys in increasing order
                    key[i] = k;
                    i++;
                }
            }
            // try to get weight an return it. failing that, register missing key
            // note that find either returns an iterator to the correct entry, or to end if the entry does not exist
            auto map_itr = w.find({key[0], key[1]});
            if (map_itr != w.end()) {
                return map_itr->second;
            }
            else {
                missing_keys.insert({key[0], key[1]});
                return 0;
            }
        }
        else if (num_bits == 3) {
            // read positions (key) from bitkey
            int i = 0;
            std::array<int, 3> key;
            for (int k = 0; k < 8 * sizeof(int); k++) {
                if ((bitkey >> k) & 1) {
                    key[i] = k;
                    i++;
                }
            }
            // try to get weight for all 3 pairs of positions, taking its maximum
            // any failure registers a missing key
            ScalarType val = -std::numeric_limits<ScalarType>::max();
            // first pair
            auto map_itr = w.find({key[0], key[1]});
            if (map_itr != w.end()) {
                val = std::max(val, map_itr->second);
            }
            else {
                missing_keys.insert({key[0], key[1]});
            }
            // second pair
            map_itr = w.find({key[1], key[2]});
            if (map_itr != w.end()) {
                val = std::max(val, map_itr->second);
            }
            else {
                missing_keys.insert({key[1], key[2]});
            }
            // third pair
            map_itr = w.find({key[0], key[2]});
            if (map_itr != w.end()) {
                val = std::max(val, map_itr->second);
            }
            else {
                missing_keys.insert({key[0], key[2]});
            }
            // return the maximum weight, or 0 by default
            return std::max(0.0, val);
        }
        else if (num_bits == 0) { // handle non-boundary points
            return 0;
        }
        else { // warn and abort if 1 or more than 3 bits are set
            mio::log(mio::LogLevel::critical, "Number of bits set should be 2 or 3, was {}.\n", num_bits);
            exit(EXIT_FAILURE);
        }
    }
};

struct FittingFunctionSetup {
    using Model  = mio::mpm::ABM<PotentialGermany<States>>;
    using Status = Model::Status;

    const Eigen::MatrixXd& potential; // reference to a WeightedPotential.potential
    const Eigen::MatrixXi metaregions;
    Eigen::MatrixXd reference_commuters; // number of commuters between regions
    double t_max;
    std::vector<Model::Agent> agents;

    const std::set<std::pair<int, int>>& border_pairs;

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
    explicit FittingFunctionSetup(const WeightedPotential& wp, const std::string& metaregions_pgm_file,
                                  const std::string& mobility_data_directory, const double t_max)
        : potential(wp.potential)
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
        , border_pairs(wp.get_keys())
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
    Model model(ffs.agents, {}, ffs.potential, ffs.metaregions, {}, sigma);

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
    WeightedPotential wp("potentially_germany.pgm", "boundary_ids.pgm");
    TIME_TYPE post_potential = TIME_NOW;
    fprintf(stdout, "# Time for creating weighted potential: %.*g\n", PRECISION,
            PRINTABLE_TIME(post_potential - pre_potential));
    TIME_TYPE pre_fitting_function_setup = TIME_NOW;
    FittingFunctionSetup ffs(wp, "metagermany.pgm", "data/mobility/", 100);
    TIME_TYPE post_fitting_function_setup = TIME_NOW;
    fprintf(stdout, "# Time for fitting function setup: %.*g\n", PRECISION,
            PRINTABLE_TIME(post_fitting_function_setup - pre_fitting_function_setup));

    TIME_TYPE pre_min_function = TIME_NOW;
    auto result                = dlib::find_min_global(
        [&](auto&& w1, auto&& w2, auto&& w3, auto&& w4, auto&& w5, auto&& w6, auto&& w7, auto&& w8, auto&& w9,
            auto&& w10, auto&& w11, auto&& w12, auto&& w13, auto&& w14, auto&& sigma1, auto&& sigma2, auto&& sigma3,
            auto&& sigma4, auto&& sigma5, auto&& sigma6, auto&& sigma7, auto&& sigma8) {
            // let dlib set the weights for the potential
            wp.apply_weights({w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14});
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
