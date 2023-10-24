#ifndef WEIGHTED_GRADIENT_H_
#define WEIGHTED_GRADIENT_H_

#include "memilio/io/json_serializer.h"
#include "weighted_potential.h"

// struct to store and reapply weights to a gradient
// reads a base gradient and boundary ids from pgm files during construction. the gradient can be accessed as a
// public member, and weights are applied to this gradient using the apply_weights function
struct WeightedGradient {
    using GradientMatrix = Eigen::Matrix<Eigen::Vector2d, Eigen::Dynamic, Eigen::Dynamic>;
    const GradientMatrix base_gradient; // unweighted gradient
    GradientMatrix gradient;
    const Eigen::MatrixXi boundary_ids; // bondaries to apply weights to

private:
    std::set<std::pair<int, int>> missing_keys;

public:
    const size_t num_weights;

private:
    std::map<std::pair<int, int>, ScalarType> weight_map;

public:
    // first load base gradient and boundary ids, then find needed weight keys and set up weight_map
    WeightedGradient(const std::string& gradient_gradient_json_file, const std::string& boundary_ids_pgm_file)
        // this uses a lot of immediately invoked lambdas
        : base_gradient([&gradient_gradient_json_file] {
            // load and read json file
            auto result = mio::read_json(gradient_gradient_json_file, mio::Tag<GradientMatrix>{});
            if (!result) {
                mio::log(mio::LogLevel::critical, result.error().formatted_message());
                exit(1);
            }
            else {
                return result.value();
            }
        }())
        , gradient(base_gradient) // assign base_gradient, as we have no weights yet
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
        assert(base_gradient.cols() == boundary_ids.cols());
        assert(base_gradient.rows() == boundary_ids.rows());
    }

    void apply_weights(const std::vector<ScalarType> weights)
    {
        assert(base_gradient.cols() == gradient.cols());
        assert(base_gradient.rows() == gradient.rows());
        assert(weights.size() == num_weights);
        // assign weights to map values
        {
            size_t i = 0;
            for (auto& weight : weight_map) {
                weight.second = weights[i];
                i++;
            }
        }
        // recompute gradient
        for (Eigen::Index i = 0; i < gradient.rows(); i++) {
            for (Eigen::Index j = 0; j < gradient.cols(); j++) {
                if (boundary_ids(i, j) > 0) { // skip non-boundary entries
                    gradient(i, j) = base_gradient(i, j) * get_weight(weight_map, boundary_ids(i, j), missing_keys);
                }
            }
        }
    }

    std::set<std::pair<int, int>> get_keys()
    {
        return missing_keys;
    }
};

#endif // WEIGHTED_GRADIENT_H_
