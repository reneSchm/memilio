#ifndef WEIGHTED_POTENTIAL_H_
#define WEIGHTED_POTENTIAL_H_

#include "hybrid_paper/library/map_reader.h"
#include "memilio/config.h"
#include "memilio/utils/logging.h"
#include "memilio/math/eigen.h"

#include <fstream>
#include <map>
#include <set>

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

    std::set<std::pair<int, int>> get_keys()
    {
        return missing_keys;
    }
};

#endif // WEIGHTED_POTENTIAL_H_
