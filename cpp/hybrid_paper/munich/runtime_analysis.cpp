#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "hybrid_paper/munich/munich_setup.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "hybrid_paper/library/sensitivity_analysis.h"
#include "sensitivity_analysis_spatial_hybrid_fcts.h"
#include <cstddef>

void save_results(std::string result_file, std::vector<double>& x, std::vector<std::vector<double>> y,
                  std::vector<std::string> prefixes)
{
    auto file        = fopen(result_file.c_str(), "w");
    std::string line = "";
    for (auto p : prefixes) {
        line += p + " ";
    }
    fprintf(file, "%s\n", line.c_str());
    for (auto i = 0; i < x.size(); ++i) {
        line = std::to_string(x[i]);
        for (auto& output : y) {
            line += " " + std::to_string(output[i]);
        }
        fprintf(file, "%s\n", line.c_str());
    }
    fclose(file);
}

int main()
{
    using Status             = mio::mpm::paper::InfectionState;
    using Region             = mio::mpm::Region;
    using ABM                = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    using PDMM               = mio::mpm::PDMModel<8, Status>;
    const size_t num_regions = 8;
    const size_t num_runs    = 112;
    int scenario             = 3;
    std::string result_path  = mio::base_dir() + "cpp/outputs/time_Munich/";
    mio::set_log_level(mio::LogLevel::warn);
    switch (scenario) {
    case 0: //scaling scenario num agents (with only susceptibles)
    {
        double ppa_min = 50;
        double ppa_max = 2000;
        auto& ppa_rng  = mio::UniformDistribution<double>::get_instance();
        std::vector<double> x;
        x.reserve(num_runs);
        std::vector<double> y_ABM;
        std::vector<double> y_PDMM;
        std::vector<double> y_Hybrid;
        y_ABM.reserve(num_runs);
        y_PDMM.reserve(num_runs);
        y_Hybrid.reserve(num_runs);
        for (size_t run = 0; run < num_runs; ++run) {
            size_t ppa = static_cast<int>(ppa_rng(ppa_min, ppa_max));
            mio::mpm::paper::MunichSetup<ABM::Agent> setup(
                3., 3., 5., std::vector<double>(num_regions, 0.0), 0.1, 0.004, mio::Date(2021, 3, 1),
                {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175},
                {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562}, ppa,
                std::vector<std::vector<double>>(num_regions, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
                []() {
                    const auto fname = mio::base_dir() + "metagermany.pgm";
                    std::ifstream ifile(fname);
                    if (!ifile.is_open()) {
                        mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
                        std::abort();
                    }
                    else {
                        auto _metaregions = mio::mpm::read_pgm_raw(ifile).first;
                        ifile.close();
                        return _metaregions;
                    }
                }(),
                150., 0.1,
                []() {
                    const std::vector<int> county_ids = {233, 228, 242, 223, 238, 232, 231, 229};
                    Eigen::MatrixXd reference_commuters =
                        get_transition_matrix(mio::base_dir() + "data/mobility/").value();
                    std::vector<double> pops{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _commute_weights(8, 8);
                    _commute_weights.setZero();
                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 8; j++) {
                            if (i != j) {
                                _commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
                            }
                        }
                        _commute_weights(i, i) = pops[i] - _commute_weights.row(i).sum();
                    }
                    return _commute_weights;
                }(),
                std::vector<double>(num_regions, 10), 50);
            std::cerr << "run " << run << " na " << setup.num_agents << "\n";
            auto draw_func_abm = [](mio::mpm::paper::MunichSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_agents_status(sim);
            };
            auto draw_func_pdmm = [](mio::mpm::paper::MunichSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_pdmm_populations(sim);
            };
            ABM abm         = setup.create_abm<ABM>();
            PDMM pdmm       = setup.create_pdmm<PDMM>();
            auto res_ABM    = sensitivity_results(setup, abm, 1, draw_func_abm);
            auto res_PDMM   = sensitivity_results(setup, pdmm, 1, draw_func_pdmm);
            auto res_Hybrid = simulate_hybridization(abm, pdmm, setup, 1);
            if (res_ABM.size() != 1 || res_PDMM.size() != 1 || res_Hybrid.size() != 1) {
                mio::log_error("Outputs do not have the correct format.");
            }
            y_ABM.push_back(res_ABM[0]);
            y_PDMM.push_back(res_PDMM[0]);
            y_Hybrid.push_back(res_Hybrid[0]);
            x.push_back(setup.num_agents);
        }
        save_results(result_path + "time_sus_scaling.txt", x, std::vector<std::vector<double>>{y_ABM, y_PDMM, y_Hybrid},
                     {"na", "ABM", "PDMM", "Hybrid"});
    } break;
    case 1: //scaling according to max infected, transmissions, deaths
    {
        auto rho_dist    = mio::ParameterDistributionUniform(0.025, 0.5);
        auto E_init_dist = mio::ParameterDistributionUniform(0, 0.005);
        auto C_init_dist = mio::ParameterDistributionUniform(0, 0.005);
        auto I_init_dist = mio::ParameterDistributionUniform(0, 0.005);
        std::vector<std::vector<double>> x_ABM(3, std::vector<double>(num_runs));
        std::vector<std::vector<double>> x_PDMM(3, std::vector<double>(num_runs));
        std::vector<std::vector<double>> x_Hybrid(3, std::vector<double>(num_runs));
        std::vector<double> y_ABM(num_runs);
        std::vector<double> y_PDMM(num_runs);
        std::vector<double> y_Hybrid(num_runs);
        const double ppa = 500;
        for (size_t run = 0; run < num_runs; ++run) {
            std::cerr << "run " << run << "\n";
            double rho    = rho_dist.get_rand_sample();
            double E_init = E_init_dist.get_rand_sample();
            double C_init = C_init_dist.get_rand_sample();
            double I_init = I_init_dist.get_rand_sample();
            double S      = 1.0 - E_init - C_init - I_init;
            mio::mpm::paper::MunichSetup<ABM::Agent> setup(
                3., 3., 5., std::vector<double>(num_regions, rho), 0.1, 0.004, mio::Date(2021, 3, 1),
                {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175},
                {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562}, ppa,
                std::vector<std::vector<double>>(num_regions, {S, E_init, C_init, I_init, 0.0, 0.0}),
                []() {
                    const auto fname = mio::base_dir() + "metagermany.pgm";
                    std::ifstream ifile(fname);
                    if (!ifile.is_open()) {
                        mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
                        std::abort();
                    }
                    else {
                        auto _metaregions = mio::mpm::read_pgm_raw(ifile).first;
                        ifile.close();
                        return _metaregions;
                    }
                }(),
                150., 0.1,
                []() {
                    const std::vector<int> county_ids = {233, 228, 242, 223, 238, 232, 231, 229};
                    Eigen::MatrixXd reference_commuters =
                        get_transition_matrix(mio::base_dir() + "data/mobility/").value();
                    std::vector<double> pops{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _commute_weights(8, 8);
                    _commute_weights.setZero();
                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 8; j++) {
                            if (i != j) {
                                _commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
                            }
                        }
                        _commute_weights(i, i) = pops[i] - _commute_weights.row(i).sum();
                    }
                    return _commute_weights;
                }(),
                std::vector<double>(num_regions, 10), 50);
            auto draw_func_abm = [](mio::mpm::paper::MunichSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_agents_status(sim);
            };
            auto draw_func_pdmm = [](mio::mpm::paper::MunichSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_pdmm_populations(sim);
            };
            ABM abm   = setup.create_abm<ABM>();
            PDMM pdmm = setup.create_pdmm<PDMM>();
            //result vector need to be [max_infected, total_transmissions, deaths]
            auto res_ABM    = sensitivity_results(setup, abm, 1, draw_func_abm);
            auto res_PDMM   = sensitivity_results(setup, pdmm, 1, draw_func_pdmm);
            auto res_Hybrid = simulate_hybridization(abm, pdmm, setup, 1);
            if (res_ABM.size() != 4 || res_PDMM.size() != 4 || res_Hybrid.size() != 4) {
                mio::log_error("Outputs do not have the correct format.");
            }
            y_ABM[run]       = res_ABM[3];
            y_PDMM[run]      = res_PDMM[3];
            y_Hybrid[run]    = res_Hybrid[3];
            x_ABM[0][run]    = res_ABM[0];
            x_ABM[1][run]    = res_ABM[1];
            x_ABM[2][run]    = res_ABM[2];
            x_PDMM[0][run]   = res_PDMM[0];
            x_PDMM[1][run]   = res_PDMM[1];
            x_PDMM[2][run]   = res_PDMM[2];
            x_Hybrid[0][run] = res_Hybrid[0];
            x_Hybrid[1][run] = res_Hybrid[1];
            x_Hybrid[2][run] = res_Hybrid[2];
        }
        //save outputs
        save_results(result_path + "time_infected_transmissions_ABM.txt", y_ABM, x_ABM,
                     {"ABM_Time", "sum_Infected", "transmissions", "deaths"});
        save_results(result_path + "time_infected_transmissions_PDMM.txt", y_PDMM, x_PDMM,
                     {"PDMM_Time", "sum_Infected", "transmissions", "deaths"});
        save_results(result_path + "time_infected_transmissions_Hybrid.txt", y_Hybrid, x_Hybrid,
                     {"Hybrid_Time", "sum_Infected", "transmissions", "deaths"});
    } break;
    case 2: {
        const std::vector<double> prop_infected{0.001, 0.005, 0.01, 0.05, 0.1};
        // x value is the proportion of infected
        std::vector<double> x(num_runs * prop_infected.size());
        // y value is the runtime
        std::vector<double> y_ABM(num_runs * prop_infected.size());
        std::vector<double> y_PDMM(num_runs * prop_infected.size());
        std::vector<double> y_Hybrid(num_runs * prop_infected.size());
        const size_t ppa = 200;
        for (size_t i = 0; i < prop_infected.size(); ++i) {
            double E_init = prop_infected[i] / 3.;
            double C_init = prop_infected[i] / 3.;
            double I_init = prop_infected[i] - E_init - C_init;
            double S      = 1.0 - E_init - C_init - I_init;
            mio::mpm::paper::MunichSetup<ABM::Agent> setup(
                3., 3., 5., std::vector<double>(num_regions, 0.2), 0.1, 0.004, mio::Date(2021, 3, 1),
                {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175},
                {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562}, ppa,
                {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {S, E_init, C_init, I_init, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
                []() {
                    const auto fname = mio::base_dir() + "metagermany.pgm";
                    std::ifstream ifile(fname);
                    if (!ifile.is_open()) {
                        mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
                        std::abort();
                    }
                    else {
                        auto _metaregions = mio::mpm::read_pgm_raw(ifile).first;
                        ifile.close();
                        return _metaregions;
                    }
                }(),
                100., 0.1,
                []() {
                    const std::vector<int> county_ids = {233, 228, 242, 223, 238, 232, 231, 229};
                    Eigen::MatrixXd reference_commuters =
                        get_transition_matrix(mio::base_dir() + "data/mobility/").value();
                    std::vector<double> pops{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _commute_weights(8, 8);
                    _commute_weights.setZero();
                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 8; j++) {
                            if (i != j) {
                                _commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
                            }
                        }
                        _commute_weights(i, i) = pops[i] - _commute_weights.row(i).sum();
                    }
                    return _commute_weights;
                }(),
                std::vector<double>(num_regions, 10), 50);
            auto draw_func_abm = [](mio::mpm::paper::MunichSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_agents_status(sim);
            };
            auto draw_func_pdmm = [](mio::mpm::paper::MunichSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_pdmm_populations(sim);
            };
            ABM abm                  = setup.create_abm<ABM>();
            setup.transmission_rates = std::vector<double>(8, 0.19);
            setup.reset_adoption_rates();
            PDMM pdmm       = setup.create_pdmm<PDMM>();
            auto res_ABM    = sensitivity_results(setup, abm, num_runs, draw_func_abm);
            auto res_PDMM   = sensitivity_results(setup, pdmm, num_runs, draw_func_pdmm);
            auto res_Hybrid = simulate_hybridization(abm, pdmm, setup, num_runs);
            if (res_ABM.size() != num_runs || res_PDMM.size() != num_runs || res_Hybrid.size() != num_runs) {
                mio::log_error("Outputs do not have the correct format.");
            }
            std::fill_n(x.begin() + i * num_runs, num_runs, prop_infected[i]);
            std::copy_n(res_ABM.cbegin(), num_runs, y_ABM.begin() + num_runs * i);
            std::copy_n(res_PDMM.cbegin(), num_runs, y_PDMM.begin() + num_runs * i);
            std::copy_n(res_Hybrid.cbegin(), num_runs, y_Hybrid.begin() + num_runs * i);
        }
        //save outputs
        save_results(result_path + "Scaling_initially_infected.txt", x,
                     std::vector<std::vector<double>>{y_ABM, y_PDMM, y_Hybrid},
                     {"prop_infected", "ABM", "PDMM", "Hybrid"});

    } break;
    case 3: {
        const std::vector<double> rho{0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4};
        // x value is rho
        std::vector<double> x(num_runs * rho.size());
        // y value is the runtime
        std::vector<double> y_ABM(num_runs * rho.size());
        std::vector<double> y_PDMM(num_runs * rho.size());
        std::vector<double> y_Hybrid(num_runs * rho.size());
        const size_t ppa = 200;
        for (size_t i = 0; i < rho.size(); ++i) {
            mio::mpm::paper::MunichSetup<ABM::Agent> setup(
                3., 3., 5., std::vector<double>(num_regions, rho[i]), 0.1, 0.004, mio::Date(2021, 3, 1),
                {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175},
                {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562}, ppa,
                {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {0.998, 0.0005, 0.0005, 0.001, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                 {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}},
                []() {
                    const auto fname = mio::base_dir() + "metagermany.pgm";
                    std::ifstream ifile(fname);
                    if (!ifile.is_open()) {
                        mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
                        std::abort();
                    }
                    else {
                        auto _metaregions = mio::mpm::read_pgm_raw(ifile).first;
                        ifile.close();
                        return _metaregions;
                    }
                }(),
                100., 0.1,
                []() {
                    const std::vector<int> county_ids = {233, 228, 242, 223, 238, 232, 231, 229};
                    Eigen::MatrixXd reference_commuters =
                        get_transition_matrix(mio::base_dir() + "data/mobility/").value();
                    std::vector<double> pops{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
                    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _commute_weights(8, 8);
                    _commute_weights.setZero();
                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 8; j++) {
                            if (i != j) {
                                _commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
                            }
                        }
                        _commute_weights(i, i) = pops[i] - _commute_weights.row(i).sum();
                    }
                    return _commute_weights;
                }(),
                std::vector<double>(num_regions, 10), 50);
            auto draw_func_abm = [](mio::mpm::paper::MunichSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_agents_status(sim);
            };
            auto draw_func_pdmm = [](mio::mpm::paper::MunichSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_pdmm_populations(sim);
            };
            ABM abm         = setup.create_abm<ABM>();
            PDMM pdmm       = setup.create_pdmm<PDMM>();
            auto res_ABM    = sensitivity_results(setup, abm, num_runs, draw_func_abm);
            auto res_PDMM   = sensitivity_results(setup, pdmm, num_runs, draw_func_pdmm);
            auto res_Hybrid = simulate_hybridization(abm, pdmm, setup, num_runs);
            if (res_ABM.size() != num_runs || res_PDMM.size() != num_runs || res_Hybrid.size() != num_runs) {
                mio::log_error("Outputs do not have the correct format.");
            }
            std::fill_n(x.begin() + i * num_runs, num_runs, rho[i]);
            std::copy_n(res_ABM.cbegin(), num_runs, y_ABM.begin() + num_runs * i);
            std::copy_n(res_PDMM.cbegin(), num_runs, y_PDMM.begin() + num_runs * i);
            std::copy_n(res_Hybrid.cbegin(), num_runs, y_Hybrid.begin() + num_runs * i);
        }
        //save outputs
        save_results(result_path + "Scaling_rho.txt", x, std::vector<std::vector<double>>{y_ABM, y_PDMM, y_Hybrid},
                     {"rho", "ABM", "PDMM", "Hybrid"});
    } break;
    default:
        break;
    }
}
