#ifndef SENSITIVITY_ANALYSIS_SETUP_QW_H
#define SENSITIVITY_ANALYSIS_SETUP_QW_H

#include "memilio/utils/parameter_distributions.h"

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

struct SensitivitySetupQW {
    const std::map<std::string, mio::ParameterDistributionUniform> params;
    const std::map<std::string, double> base_values;
    const std::map<std::string, double> deltas;
    std::vector<std::map<std::string, std::vector<double>>> diffs;
    std::vector<std::map<std::string, std::vector<double>>> rel_effects;

    SensitivitySetupQW(std::vector<double> t_Exposed_range, std::vector<double> t_Carrier_range,
                       std::vector<double> t_Infected_range, std::vector<double> mu_C_R_range,
                       std::vector<double> mu_I_D_range, std::vector<double> transmission_rate_range,
                       std::vector<double> sigma_range, std::vector<double> contact_radius_range,
                       std::vector<double> E_init_range, std::vector<double> C_init_range,
                       std::vector<double> I_init_range, std::vector<double> transition_rates_range, size_t num_runs,
                       size_t num_outputs)
        : params(
              {{"t_Exposed", mio::ParameterDistributionUniform(t_Exposed_range[0], t_Exposed_range[1])},
               {"t_Carrier", mio::ParameterDistributionUniform(t_Carrier_range[0], t_Carrier_range[1])},
               {"t_Infected", mio::ParameterDistributionUniform(t_Infected_range[0], t_Infected_range[1])},
               {"transmission_rate",
                mio::ParameterDistributionUniform(transmission_rate_range[0], transmission_rate_range[1])},
               {"mu_C_R", mio::ParameterDistributionUniform(mu_C_R_range[0], mu_C_R_range[1])},
               {"mu_I_D", mio::ParameterDistributionUniform(mu_I_D_range[0], mu_I_D_range[1])},
               {"sigma", mio::ParameterDistributionUniform(sigma_range[0], sigma_range[1])},
               {"contact_radius", mio::ParameterDistributionUniform(contact_radius_range[0], contact_radius_range[1])},
               {"E", mio::ParameterDistributionUniform(E_init_range[0], E_init_range[1])},
               {"C", mio::ParameterDistributionUniform(C_init_range[0], C_init_range[1])},
               {"I", mio::ParameterDistributionUniform(I_init_range[0], I_init_range[1])},
               {"transition_rates",
                mio::ParameterDistributionUniform(transition_rates_range[0], transition_rates_range[1])},
               {"dummy", mio::ParameterDistributionUniform(sigma_range[0], sigma_range[1])}})
        , deltas({{"t_Exposed", (t_Exposed_range[1] - t_Exposed_range[0]) / 7.},
                  {"t_Carrier", (t_Carrier_range[1] - t_Carrier_range[0]) / 7.},
                  {"t_Infected", (t_Infected_range[1] - t_Infected_range[0]) / 7.},
                  {"transmission_rate", (transmission_rate_range[1] - transmission_rate_range[0]) / 7.},
                  {"mu_C_R", (mu_C_R_range[1] - mu_C_R_range[0]) / 7.},
                  {"mu_I_D", (mu_I_D_range[1] - mu_I_D_range[0]) / 7.},
                  {"sigma", ((sigma_range[1] - sigma_range[0]) / 7.)},
                  {"contact_radius", (contact_radius_range[1] - contact_radius_range[0]) / 7.},
                  {"E", (E_init_range[1] - E_init_range[0]) / 7.},
                  {"C", (C_init_range[1] - C_init_range[0]) / 7.},
                  {"I", (I_init_range[1] - I_init_range[0]) / 7.},
                  {"transition_rates", (transition_rates_range[1] - transition_rates_range[0]) / 7.},
                  {"dummy", ((sigma_range[1] - sigma_range[0]) / 7.)}})
        , base_values({{"t_Exposed", 0.0},
                       {"t_Carrier", 0.0},
                       {"t_Infected", 0.0},
                       {"transmission_rate", 0.0},
                       {"mu_C_R", 0.0},
                       {"mu_I_D", 0.0},
                       {"sigma", 0.0},
                       {"contact_radius", 0.0},
                       {"E", 0.0},
                       {"C", 0.0},
                       {"I", 0.0},
                       {"transition_rates", 0.0},
                       {"dummy", 0.0}})
    {
        diffs = std::vector<std::map<std::string, std::vector<double>>>(
            num_outputs, {{"t_Exposed", std::vector<double>(num_runs)},
                          {"t_Carrier", std::vector<double>(num_runs)},
                          {"t_Infected", std::vector<double>(num_runs)},
                          {"transmission_rate", std::vector<double>(num_runs)},
                          {"mu_C_R", std::vector<double>(num_runs)},
                          {"mu_I_D", std::vector<double>(num_runs)},
                          {"sigma", std::vector<double>(num_runs)},
                          {"contact_radius", std::vector<double>(num_runs)},
                          {"E", std::vector<double>(num_runs)},
                          {"C", std::vector<double>(num_runs)},
                          {"I", std::vector<double>(num_runs)},
                          {"transition_rates", std::vector<double>(num_runs)},
                          {"dummy", std::vector<double>(num_runs)}});
        rel_effects = std::vector<std::map<std::string, std::vector<double>>>(
            num_outputs, {{"t_Exposed", std::vector<double>(num_runs)},
                          {"t_Carrier", std::vector<double>(num_runs)},
                          {"t_Infected", std::vector<double>(num_runs)},
                          {"transmission_rate", std::vector<double>(num_runs)},
                          {"mu_C_R", std::vector<double>(num_runs)},
                          {"mu_I_D", std::vector<double>(num_runs)},
                          {"sigma", std::vector<double>(num_runs)},
                          {"contact_radius", std::vector<double>(num_runs)},
                          {"E", std::vector<double>(num_runs)},
                          {"C", std::vector<double>(num_runs)},
                          {"I", std::vector<double>(num_runs)},
                          {"transition_rates", std::vector<double>(num_runs)},
                          {"dummy", std::vector<double>(num_runs)}});
    }

    SensitivitySetupQW(size_t num_runs, size_t num_outputs)
        : SensitivitySetupQW({2.67, 4.}, //t_Exposed
                             {1.2, 6.73}, //t_Carrier
                             {5., 12.}, //t_Infected
                             {0.15, 0.3}, //mu_C_R
                             {0.0, 0.08}, //mu_I_D
                             {0.025, 0.6}, //transmission_rate
                             {0.4, 0.6}, //sigma
                             {0.05, 0.6}, //contact_radius
                             {0.0, 0.01}, //E
                             {0.0, 0.01}, //C
                             {0.0, 0.01}, //I
                             {0.00002, 0.0125}, //transition_rates
                             num_runs, num_outputs)
    {
    }
};
#endif //SENSITIVITY_ANALYSIS_SETUP_QW_H
