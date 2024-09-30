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
    std::vector<std::map<std::string, std::vector<double>>> elem_effects;

    SensitivitySetupQW(std::vector<double> t_Exposed_range, std::vector<double> mu_C_R_range,
                       std::vector<double> mu_I_D_range, std::vector<double> transmission_rate_range,
                       std::vector<double> sigma_range, std::vector<double> contact_radius_range,
                       std::vector<double> E_init_range, std::vector<double> C_init_range,
                       std::vector<double> I_init_range, std::vector<double> transition_rates_range, size_t num_runs,
                       size_t num_outputs)
        : params(
              {{"t_Exposed", mio::ParameterDistributionUniform(t_Exposed_range[0], t_Exposed_range[1])},
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
                mio::ParameterDistributionUniform(transition_rates_range[0], transition_rates_range[1])}})
        , deltas({{"t_Exposed", (t_Exposed_range[1] - t_Exposed_range[0]) / 10.},
                  {"transmission_rate", (transmission_rate_range[1] - transmission_rate_range[0]) / 10.},
                  {"mu_C_R", (mu_C_R_range[1] - mu_C_R_range[0]) / 10.},
                  {"mu_I_D", (mu_I_D_range[1] - mu_I_D_range[0]) / 10.},
                  {"sigma", ((sigma_range[1] - sigma_range[0]) / 10.)},
                  {"contact_radius", (contact_radius_range[1] - contact_radius_range[0]) / 10.},
                  {"E", (E_init_range[1] - E_init_range[0]) / 10.},
                  {"C", (C_init_range[1] - C_init_range[0]) / 10.},
                  {"I", (I_init_range[1] - I_init_range[0]) / 10.},
                  {"transition_rates", (transition_rates_range[1] - transition_rates_range[0]) / 10.}})
        , base_values({{"t_Exposed", 0.0},
                       {"transmission_rate", 0.0},
                       {"mu_C_R", 0.0},
                       {"mu_I_D", 0.0},
                       {"sigma", 0.0},
                       {"contact_radius", 0.0},
                       {"E", 0.0},
                       {"C", 0.0},
                       {"I", 0.0},
                       {"transition_rates", 0.0}})
    {
        elem_effects = std::vector<std::map<std::string, std::vector<double>>>(
            num_outputs, {{"t_Exposed", std::vector<double>(num_runs)},
                          {"transmission_rate", std::vector<double>(num_runs)},
                          {"mu_C_R", std::vector<double>(num_runs)},
                          {"mu_I_D", std::vector<double>(num_runs)},
                          {"sigma", std::vector<double>(num_runs)},
                          {"contact_radius", std::vector<double>(num_runs)},
                          {"E", std::vector<double>(num_runs)},
                          {"C", std::vector<double>(num_runs)},
                          {"I", std::vector<double>(num_runs)},
                          {"transition_rates", std::vector<double>(num_runs)}});
    }

    SensitivitySetupQW(size_t num_runs, size_t num_outputs)
        : SensitivitySetupQW({1.5, 5.}, //t_Exposed = 1/gamma_E_C
                             {0.01, 0.5}, //mu_C_R = gamma_C_R
                             {0.0001, 0.1}, //mu_I_D = gamma_I_D
                             {0.01, 0.6}, //transmission_rate
                             {0.1, 0.6}, //sigma
                             {0.1, 1.}, //contact_radius
                             {0.0, 0.1}, //E
                             {0.0, 0.1}, //C
                             {0.0, 0.1}, //I
                             {0.00001, 0.05}, //transition_rates
                             num_runs, num_outputs)
    {
    }
};
#endif //SENSITIVITY_ANALYSIS_SETUP_QW_H
