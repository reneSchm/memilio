#include "cpp/models/mpm/potentials/map_reader.h"
#include "cpp/memilio/io/cli.h"
#include <map>

// use the following command to compile
// g++ --std=c++17 -Wall --pedantic -Icpp -Icpp/build/memilio -Icpp/build/_deps/eigen-src -Icpp/build/_deps/jsoncpp-src/include/ -Wl,-rpath,$(pwd)/cpp/build/lib map_apply_border_weights.cpp cpp/build/lib/libjsoncpp.so cpp/models/mpm/potentials/map_reader.cpp

using namespace mio::mpm;

struct Weights {
    const static std::string name()
    {
        return "Weights";
    }
    const static std::string alias()
    {
        return "w";
    }
    using Type = std::vector<std::vector<ScalarType>>;
};

std::map<std::pair<int, int>, int> missing_keys;

// Return the (maximum) weight corresponding to (all) postions in bitkey.
// Accepts bitkeys with 0, 2 or 3 bits set. E.g., the third bit corresponds to position 3.
// Positions are requested from map as pair (a, b) where a < b. If no entry is present, returns 0.
ScalarType get_weight(const std::map<std::pair<int, int>, ScalarType>& w, int bitkey)
{
    auto num_bits = num_bits_set(bitkey);
    if (num_bits == 2) { // border between two positions
        // read positions (key) from bitkey
        int i = 0;
        std::array<int, 2> key;
        for (int k = 0; k < 16; k++) {
            if ((bitkey >> k) & 1) {
                key[i] = k;
                i++;
            }
        }
        // try to get weight. failing that, register missing key
        auto map_itr = w.find({key[0], key[1]});
        if (map_itr != w.end()) {
            return map_itr->second;
        }
        else {
            missing_keys[{key[0], key[1]}] = 0;
            return 0;
        }
    }
    else if (num_bits == 3) {
        // read positions (key) from bitkey
        int i = 0;
        std::array<int, 3> key;
        for (int k = 0; k < 16; k++) {
            if ((bitkey >> k) & 1) {
                key[i] = k;
                i++;
            }
        }
        // try to get weight for all 3 pairs of positions
        // any failure registers a missing key
        ScalarType val = -std::numeric_limits<ScalarType>::max();
        auto map_itr   = w.find({key[0], key[1]});
        if (map_itr != w.end()) {
            val = std::max(val, map_itr->second);
        }
        else {
            missing_keys[{key[0], key[1]}] = 0;
        }
        map_itr = w.find({key[1], key[2]});
        if (map_itr != w.end()) {
            val = std::max(val, map_itr->second);
        }
        else {
            missing_keys[{key[1], key[2]}] = 0;
        }
        map_itr = w.find({key[0], key[2]});
        if (map_itr != w.end()) {
            val = std::max(val, map_itr->second);
        }
        else {
            missing_keys[{key[0], key[2]}] = 0;
        }
        return std::max(0.0, val);
    }
    else if (num_bits == 0) {
        return 0;
    }
    else {
        std::cerr << "Number of bits set should be 2 or 3, was " << num_bits << "\n";
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char** argv)
{
    // allows setting the weights via cli
    mio::ParameterSet<Weights> p;
    p.get<Weights>().push_back({5, 9, 2.0});
    auto result = mio::command_line_interface(argv[0], argc, argv, p);
    if (!result) {
        std::cout << result.error().formatted_message();
        return result.error().code().value();
    }
    auto& w = p.get<Weights>();
    // enter the given weights into a map
    std::map<std::pair<int, int>, ScalarType> weights;
    for (auto weight : w) {
        weights[{(int)weight[0], (int)weight[1]}] = weight[2];
    }
    // open the potential and bitkey files
    std::fstream ifile;
    ifile.open("potentially_germany.pgm");
    Eigen::MatrixXd potential = read_pgm(ifile);
    ifile.close();
    ifile.open("boundary_ids.pgm");
    Eigen::MatrixXi bitkeys = read_pgm_raw(ifile).first;
    ifile.close();
    assert(potential.cols() == bitkeys.cols());
    assert(potential.rows() == bitkeys.rows());
    // scale potential by the above weights
    for (Eigen::Index i = 0; i < potential.rows(); i++) {
        for (Eigen::Index j = 0; j < potential.cols(); j++) {
            if (bitkeys(i, j) > 0) // skip non-boundary entries
                potential(i, j) = potential(i, j) * get_weight(weights, bitkeys(i, j));
        }
    }
    // write out weighted potential
    ifile.open("weighted_germany.pgm");
    write_pgm(ifile, potential);
    ifile.close();
    // echo missing map entries
    for (auto m : missing_keys) {
        std::cerr << "Missing weight for key pair <" << m.first.first << ", " << m.first.second << ">\n";
    }
}