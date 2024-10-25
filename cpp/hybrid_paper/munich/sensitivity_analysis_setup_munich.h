#ifndef SENSITIVITY_ANALYSIS_SETUP_MUNICH_H
#define SENSITIVITY_ANALYSIS_SETUP_MUNICH_H

#include "hybrid_paper/quad_well/sensitivity_analysis_setup_qw.h"
#include "memilio/utils/parameter_distributions.h"
#include <cstddef>
#include <vector>

struct SensitivitySetupMunich {
    const std::map<std::string, mio::ParameterDistributionUniform> params;
    const std::map<std::string, double> deltas;
    const std::map<std::string, double> base_values;
    std::vector<std::map<std::string, std::vector<double>>> diffs;
    std::vector<std::map<std::string, std::vector<double>>> rel_effects;

    SensitivitySetupMunich(std::vector<double> t_Exposed_range, std::vector<double> t_Carrier_range,
                           std::vector<double> t_Infected_range, std::vector<double> transmission_rate_range,
                           std::vector<double> mu_C_R_range, std::vector<double> mu_I_D_range,
                           std::vector<double> E_init_range, std::vector<double> C_init_range,
                           std::vector<double> I_init_range, std::vector<double> commute_weights_range,
                           std::vector<double> sigma_range, std::vector<double> contact_radius_range, size_t num_runs,
                           size_t num_outputs)
        : params(
              {{"t_Exposed", mio::ParameterDistributionUniform(t_Exposed_range[0], t_Exposed_range[1])},
               {"t_Carrier", mio::ParameterDistributionUniform(t_Carrier_range[0], t_Carrier_range[1])},
               {"t_Infected", mio::ParameterDistributionUniform(t_Infected_range[0], t_Infected_range[1])},
               {"transmission_rate",
                mio::ParameterDistributionUniform(transmission_rate_range[0], transmission_rate_range[1])},
               {"mu_C_R", mio::ParameterDistributionUniform(mu_C_R_range[0], mu_C_R_range[1])},
               {"mu_I_D", mio::ParameterDistributionUniform(mu_I_D_range[0], mu_I_D_range[1])},
               {"E", mio::ParameterDistributionUniform(E_init_range[0], E_init_range[1])},
               {"C", mio::ParameterDistributionUniform(C_init_range[0], C_init_range[1])},
               {"I", mio::ParameterDistributionUniform(I_init_range[0], I_init_range[1])},
               {"commute_weights",
                mio::ParameterDistributionUniform(commute_weights_range[0], commute_weights_range[1])},
               {"sigma", mio::ParameterDistributionUniform(sigma_range[0], sigma_range[1])},
               {"contact_radius", mio::ParameterDistributionUniform(contact_radius_range[0], contact_radius_range[1])},
               {"dummy1", mio::ParameterDistributionUniform(mu_I_D_range[0], mu_I_D_range[1])},
               {"dummy2", mio::ParameterDistributionUniform(commute_weights_range[0], commute_weights_range[1])}})
        , deltas({{"t_Exposed", (t_Exposed_range[1] - t_Exposed_range[0]) / 7.},
                  {"t_Carrier", (t_Carrier_range[1] - t_Carrier_range[0]) / 7.},
                  {"t_Infected", (t_Infected_range[1] - t_Infected_range[0]) / 7.},
                  {"transmission_rate", (transmission_rate_range[1], transmission_rate_range[0]) / 7.},
                  {"mu_C_R", (mu_C_R_range[1] - mu_C_R_range[0]) / 7.},
                  {"mu_I_D", (mu_I_D_range[1] - mu_I_D_range[0]) / 7.},
                  {"E", (E_init_range[1] - E_init_range[0]) / 7.},
                  {"C", (C_init_range[1] - C_init_range[0]) / 7.},
                  {"I", (I_init_range[1] - I_init_range[0]) / 7.},
                  {"commute_weights", (commute_weights_range[1] - commute_weights_range[0]) / 7.},
                  {"sigma", (sigma_range[1] - sigma_range[0]) / 7.},
                  {"contact_radius", (contact_radius_range[1] - contact_radius_range[0]) / 7.},
                  {"dummy1", (mu_I_D_range[1] - mu_I_D_range[0]) / 7.},
                  {"dummy2", (commute_weights_range[1] - commute_weights_range[0]) / 7.}})
        , base_values({{"t_Exposed", 0.0},
                       {"t_Carrier", 0.0},
                       {"t_Infected", 0.0},
                       {"transmission_rate", 0.0},
                       {"mu_C_R", 0.0},
                       {"mu_I_D", 0.0},
                       {"E", 0.0},
                       {"C", 0.0},
                       {"I", 0.0},
                       {"commute_weights", 0.0},
                       {"sigma", 0.0},
                       {"contact_radius", 0.0},
                       {"dummy1", 0.0},
                       {"dummy2", 0.0}})
    {
        diffs = std::vector<std::map<std::string, std::vector<double>>>(
            num_outputs, {{"t_Exposed", std::vector<double>(num_runs)},
                          {"t_Carrier", std::vector<double>(num_runs)},
                          {"t_Infected", std::vector<double>(num_runs)},
                          {"transmission_rate", std::vector<double>(num_runs)},
                          {"mu_C_R", std::vector<double>(num_runs)},
                          {"mu_I_D", std::vector<double>(num_runs)},
                          {"E", std::vector<double>(num_runs)},
                          {"C", std::vector<double>(num_runs)},
                          {"I", std::vector<double>(num_runs)},
                          {"commute_weights", std::vector<double>(num_runs)},
                          {"sigma", std::vector<double>(num_runs)},
                          {"contact_radius", std::vector<double>(num_runs)},
                          {"dummy1", std::vector<double>(num_runs)},
                          {"dummy2", std::vector<double>(num_runs)}});
        rel_effects = std::vector<std::map<std::string, std::vector<double>>>(
            num_outputs, {{"t_Exposed", std::vector<double>(num_runs)},
                          {"t_Carrier", std::vector<double>(num_runs)},
                          {"t_Infected", std::vector<double>(num_runs)},
                          {"transmission_rate", std::vector<double>(num_runs)},
                          {"mu_C_R", std::vector<double>(num_runs)},
                          {"mu_I_D", std::vector<double>(num_runs)},
                          {"E", std::vector<double>(num_runs)},
                          {"C", std::vector<double>(num_runs)},
                          {"I", std::vector<double>(num_runs)},
                          {"commute_weights", std::vector<double>(num_runs)},
                          {"sigma", std::vector<double>(num_runs)},
                          {"contact_radius", std::vector<double>(num_runs)},
                          {"dummy1", std::vector<double>(num_runs)},
                          {"dummy2", std::vector<double>(num_runs)}});
    }

    SensitivitySetupMunich(size_t num_runs, size_t num_outputs)
        : SensitivitySetupMunich({2.67, 4.}, //t_Exposed
                                 {1.2, 6.73}, //t_Carrier
                                 {5., 12.}, //t_Infected
                                 {0.025, 0.6}, //transmission_rate
                                 {0.15, 0.3}, //mu_C_R
                                 {0.0, 0.08}, //mu_I_D
                                 {0.0, 0.01}, //E,
                                 {0.0, 0.01}, //C,
                                 {0.0, 0.01}, //I
                                 {0.001, 0.22}, //commute weights
                                 {5., 15.}, //sigma
                                 {5., 60.}, //contact radius
                                 num_runs, num_outputs)
    {
    }
};

#endif //SENSITIVITY_ANALYSIS_SETUP_MUNICH_H
