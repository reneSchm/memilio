#include "sensitivity_analysis.h"
#include "hybrid_paper/library/infection_state.h"
#include <cstddef>
#include <numeric>
#include <string>
#include <vector>

template <>
QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>
create_model_setup<QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>>(
    std::map<std::string, double>& params, double tmax, double dt, size_t num_agents)
{
    using Status = mio::mpm::paper::InfectionState;
    double S     = 1 - params.at("E") - params.at("C") - params.at("I");
    std::map<std::tuple<Status, mio::mpm::Region, mio::mpm::Region>, double> tr_map;
    // add transitions are between wells 0<->1, 0<->2, 1<->3 and 2<->3
    std::vector<std::tuple<int, int>> trans_pairs{{0, 1}, {0, 2}, {1, 3}, {2, 3}};
    for (auto& pair : trans_pairs) {
        for (int s = 0; s < int(Status::Count); ++s) {
            tr_map.insert({{Status(s), mio::mpm::Region(std::get<0>(pair)), mio::mpm::Region(std::get<1>(pair))},
                           params.at("transition_rates")});
            tr_map.insert({{Status(s), mio::mpm::Region(std::get<1>(pair)), mio::mpm::Region(std::get<0>(pair))},
                           params.at("transition_rates")});
        }
    }
    QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent> setup(
        params.at("t_Exposed"), params.at("t_Carrier"), params.at("t_Infected"),
        std::vector<double>{params.at("transmission_rate"), params.at("transmission_rate"),
                            params.at("transmission_rate"), params.at("transmission_rate")},
        params.at("mu_C_R"), params.at("mu_I_D"), tmax, dt, params.at("sigma"), params.at("contact_radius"), num_agents,
        std::vector<std::vector<double>>{{S, params.at("E"), params.at("C"), params.at("I"), 0.0, 0.0},
                                         {S, params.at("E"), params.at("C"), params.at("I"), 0.0, 0.0},
                                         {S, params.at("E"), params.at("C"), params.at("I"), 0.0, 0.0},
                                         {S, params.at("E"), params.at("C"), params.at("I"), 0.0, 0.0}},
        tr_map);
    return setup;
}

template <>
mio::mpm::paper::MunichSetup<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>::Agent> create_model_setup<
    mio::mpm::paper::MunichSetup<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>::Agent>>(
    std::map<std::string, double>& params, double tmax, double dt, size_t num_agents)
{
    // potential
    const auto fname = mio::base_dir() + "metagermany.pgm";
    std::ifstream ifile(fname);
    if (!ifile.is_open()) {
        mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
        std::abort();
    }
    auto metaregions = mio::mpm::read_pgm_raw(ifile).first;
    ifile.close();

    // commute weights
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(8, 8);
    commute_weights.setZero();
    for (int i = 0; i < commute_weights.rows(); i++) {
        for (int j = 0; j < commute_weights.cols(); j++) {
            if (i != j) {
                commute_weights(i, j) = params.at("commute_weights") * populations[i];
            }
        }
        commute_weights(i, i) = populations[i] - commute_weights.row(i).sum();
    }

    double S = 1 - params.at("E") - params.at("C") - params.at("I");

    mio::mpm::paper::MunichSetup<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>::Agent> setup(
        params.at("t_Exposed"), params.at("t_Carrier"), params.at("t_Infected"),
        std::vector<double>(8, params.at("transmission_rate")), params.at("mu_C_R"), params.at("mu_I_D"),
        mio::Date(2021, 3, 1), {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175}, populations,
        int(std::accumulate(populations.begin(), populations.end(), 0.0) / num_agents),
        std::vector<std::vector<double>>(8, {S, params.at("E"), params.at("C"), params.at("I"), 0.0, 0.0}), metaregions,
        tmax, dt, commute_weights, std::vector<double>(8, params.at("sigma")), params.at("contact_radius"));
    return setup;
}

template <>
mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>
create_model<QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>,
             mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>>(
    const QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>& model_setup)
{
    return model_setup.create_abm<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>>();
}

template <>
mio::mpm::PDMModel<4, mio::mpm::paper::InfectionState>
create_model<QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>,
             mio::mpm::PDMModel<4, mio::mpm::paper::InfectionState>>(
    const QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>& model_setup)
{
    return model_setup.create_pdmm<mio::mpm::PDMModel<4, mio::mpm::paper::InfectionState>>();
}

template <>
mio::mpm::ABM<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>>
create_model<mio::mpm::paper::MunichSetup<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>::Agent>,
             mio::mpm::ABM<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>>>(
    const mio::mpm::paper::MunichSetup<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>::Agent>&
        model_setup)
{
    return model_setup.create_abm<mio::mpm::ABM<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>>>();
}

template <>
mio::mpm::PDMModel<8, mio::mpm::paper::InfectionState>
create_model<mio::mpm::paper::MunichSetup<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>::Agent>,
             mio::mpm::PDMModel<8, mio::mpm::paper::InfectionState>>(
    const mio::mpm::paper::MunichSetup<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>::Agent>&
        model_setup)
{
    return model_setup.create_pdmm<mio::mpm::PDMModel<8, mio::mpm::paper::InfectionState>>();
}

void save_elementary_effects(std::vector<std::map<std::string, std::vector<double>>>& elem_effects,
                             std::string result_file, size_t num_runs)
{
    for (size_t i = 0; i < elem_effects.size(); ++i) {
        std::string filename = result_file + std::to_string(i) + ".txt";
        auto file            = fopen(filename.c_str(), "w");
        std::string line     = "";
        for (auto& elem : elem_effects[i]) {
            line += elem.first + " ";
        }
        fprintf(file, "%s\n", line.c_str());
        for (size_t run = 0; run < num_runs; ++run) {
            for (auto& elem : elem_effects[i]) {
                fprintf(file, "%f ", elem.second[run]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }
}

void draw_base_values(std::map<std::string, mio::ParameterDistributionUniform>& params,
                      std::map<std::string, double>& base_values)
{
    for (auto it = base_values.begin(); it != base_values.end(); ++it) {
        it->second = params.at(it->first).get_rand_sample();
    }
}
