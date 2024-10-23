#include "hybrid_paper/library/sensitivity_analysis.h"
#include "hybrid_paper/quad_well/quad_well_setup.h"
#include "memilio/utils/logging.h"
#include "sensitivity_analysis_spatial_hybrid_fcts.h"
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

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
    using ABM                = mio::mpm::ABM<QuadWellModel<Status>>;
    using PDMM               = mio::mpm::PDMModel<4, Status>;
    const size_t num_regions = 4;
    const size_t num_runs    = 50;
    int scenario             = 1;
    std::string result_path  = "cpp/outputs/QuadWell/time_measure/20241022_v1/";
    mio::set_log_level(mio::LogLevel::warn);
    switch (scenario) {
    case 0: //Scaling only with susceptibles according to num_agents
    {
        //TODO: read pos ausschalten
        double na_min = 80;
        double na_max = 30000;
        //std::vector<double> num_agents{80, 400, 800, 1000, 4000, 8000, 12000}; //, 16000, 20000, 30000, 40000
        auto& na_rng = mio::UniformDistribution<double>::get_instance();
        std::vector<double> x;
        x.reserve(num_runs);
        std::vector<double> y_ABM;
        std::vector<double> y_PDMM;
        std::vector<double> y_Hybrid;
        y_ABM.reserve(num_runs);
        y_PDMM.reserve(num_runs);
        y_Hybrid.reserve(num_runs);
        for (size_t run = 0; run < num_runs; ++run) {
            //size_t na = num_agents[na_rng(std::vector<double>(num_agents.size(), 1.0))];
            size_t na = static_cast<size_t>(na_rng(na_min, na_max));
            std::cerr << "run " << run << " na " << na << "\n";
            QuadWellSetup<ABM::Agent> setup(
                3.0, 3.0, 5.0, std::vector<double>(num_regions, 0.0), 0.1, 0.004, 150.0, 0.1, 0.55, 0.4, na,
                std::vector<std::vector<double>>(num_regions, {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
                std::map<std::tuple<Status, mio::mpm::Region, mio::mpm::Region>, double>{
                    {{Status::S, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0044},
                    {{Status::S, mio::mpm::Region(1), mio::mpm::Region(0)}, 0.0044},
                    {{Status::S, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0044},
                    {{Status::S, mio::mpm::Region(2), mio::mpm::Region(0)}, 0.0044},
                    {{Status::S, mio::mpm::Region(0), mio::mpm::Region(3)}, 1e-07},
                    {{Status::S, mio::mpm::Region(1), mio::mpm::Region(2)}, 1e-07},
                    {{Status::S, mio::mpm::Region(3), mio::mpm::Region(0)}, 1e-07},
                    {{Status::S, mio::mpm::Region(2), mio::mpm::Region(1)}, 1e-07},
                    {{Status::S, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0044},
                    {{Status::S, mio::mpm::Region(3), mio::mpm::Region(1)}, 0.0044},
                    {{Status::S, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0044},
                    {{Status::S, mio::mpm::Region(3), mio::mpm::Region(2)}, 0.0044}});
            auto draw_func_abm = [](QuadWellSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_agents_status(sim);
            };
            auto draw_func_pdmm = [](QuadWellSetup<ABM::Agent> setup, auto& sim) {
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
            x.push_back(na);
        }
        //save outputs
        save_results(result_path + "time_sus_scaling.txt", x, std::vector<std::vector<double>>{y_ABM, y_PDMM, y_Hybrid},
                     {"na", "ABM", "PDMM", "Hybrid"});
    } break;
    case 1: //Scaling according to sum infected and total transmissions
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
        const size_t na = 4000;
        for (size_t run = 0; run < num_runs; ++run) {
            std::cerr << "run " << run << "\n";
            double rho    = rho_dist.get_rand_sample();
            double E_init = E_init_dist.get_rand_sample();
            double C_init = C_init_dist.get_rand_sample();
            double I_init = I_init_dist.get_rand_sample();
            double S      = 1.0 - E_init - C_init - I_init;
            QuadWellSetup<ABM::Agent> setup(
                3., 3., 5., std::vector<double>(num_regions, rho), 0.1, 0.0, 150., 0.1, 0.55, 0.1, na,
                std::vector<std::vector<double>>(num_regions, {S, E_init, C_init, I_init, 0.0, 0.0}), //mu_I_D 0.004
                std::map<std::tuple<Status, mio::mpm::Region, mio::mpm::Region>, double>{
                    //transition rates for sigma = 0.55
                    {{Status::S, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0044}, //0->1
                    {{Status::E, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0044},
                    {{Status::C, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0044},
                    {{Status::I, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0044},
                    {{Status::R, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0044},
                    {{Status::S, mio::mpm::Region(1), mio::mpm::Region(0)}, 0.0044}, //1->0
                    {{Status::E, mio::mpm::Region(1), mio::mpm::Region(0)}, 0.0044},
                    {{Status::C, mio::mpm::Region(1), mio::mpm::Region(0)}, 0.0044},
                    {{Status::I, mio::mpm::Region(1), mio::mpm::Region(0)}, 0.0044},
                    {{Status::R, mio::mpm::Region(1), mio::mpm::Region(0)}, 0.0044},
                    {{Status::S, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0044}, //0->2
                    {{Status::E, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0044},
                    {{Status::C, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0044},
                    {{Status::I, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0044},
                    {{Status::R, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0044},
                    {{Status::S, mio::mpm::Region(2), mio::mpm::Region(0)}, 0.0044}, //2->0
                    {{Status::E, mio::mpm::Region(2), mio::mpm::Region(0)}, 0.0044},
                    {{Status::C, mio::mpm::Region(2), mio::mpm::Region(0)}, 0.0044},
                    {{Status::I, mio::mpm::Region(2), mio::mpm::Region(0)}, 0.0044},
                    {{Status::R, mio::mpm::Region(2), mio::mpm::Region(0)}, 0.0044},
                    {{Status::S, mio::mpm::Region(0), mio::mpm::Region(3)}, 1e-07}, //0->3 1e-07
                    {{Status::S, mio::mpm::Region(1), mio::mpm::Region(2)}, 1e-07}, //1->2
                    {{Status::S, mio::mpm::Region(3), mio::mpm::Region(0)}, 1e-07}, //3->0 1e-07
                    {{Status::S, mio::mpm::Region(2), mio::mpm::Region(1)}, 1e-07}, //2->1
                    {{Status::S, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0044}, //1->3
                    {{Status::E, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0044},
                    {{Status::C, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0044},
                    {{Status::I, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0044},
                    {{Status::R, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0044},
                    {{Status::S, mio::mpm::Region(3), mio::mpm::Region(1)}, 0.0044}, //3->1
                    {{Status::E, mio::mpm::Region(3), mio::mpm::Region(1)}, 0.0044},
                    {{Status::C, mio::mpm::Region(3), mio::mpm::Region(1)}, 0.0044},
                    {{Status::I, mio::mpm::Region(3), mio::mpm::Region(1)}, 0.0044},
                    {{Status::R, mio::mpm::Region(3), mio::mpm::Region(1)}, 0.0044},
                    {{Status::S, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0044}, //2->3
                    {{Status::E, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0044},
                    {{Status::C, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0044},
                    {{Status::I, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0044},
                    {{Status::R, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0044},
                    {{Status::S, mio::mpm::Region(3), mio::mpm::Region(2)}, 0.0044}, //3->2
                    {{Status::E, mio::mpm::Region(3), mio::mpm::Region(2)}, 0.0044},
                    {{Status::C, mio::mpm::Region(3), mio::mpm::Region(2)}, 0.0044},
                    {{Status::I, mio::mpm::Region(3), mio::mpm::Region(2)}, 0.0044},
                    {{Status::R, mio::mpm::Region(3), mio::mpm::Region(2)}, 0.0044}});
            auto draw_func_abm = [](QuadWellSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_agents_status(sim);
            };
            auto draw_func_pdmm = [](QuadWellSetup<ABM::Agent> setup, auto& sim) {
                setup.redraw_pdmm_populations(sim);
            };
            ABM abm         = setup.create_abm<ABM>();
            PDMM pdmm       = setup.create_pdmm<PDMM>();
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
    default:
        break;
    }

    return 0;
}
