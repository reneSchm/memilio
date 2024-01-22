#include "library/infection_state.h"
#include "models/mpm/region.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "library/potentials/commuting_potential.h"
#include "library/model_setup.h"
#include "library/initialization.h"
#include "memilio/utils/time_series.h"
#include "memilio/data/analyze_result.h"
#include <cstddef>
#include <memory>
#include <numeric>

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()
#define PRECISION 17

#define restart_timer(timer, description)                                                                              \
    {                                                                                                                  \
        TIME_TYPE new_time = TIME_NOW;                                                                                 \
        std::cout << "\r" << description << " :: " << PRINTABLE_TIME(new_time - timer) << std::endl << std::flush;     \
        timer = new_time;                                                                                              \
    }

std::vector<std::vector<mio::TimeSeries<double>>>
get_format_for_percentile_output(std::vector<mio::TimeSeries<double>>& ensemble_result, size_t num_regions)
{
    std::vector<std::vector<mio::TimeSeries<double>>> ensemble_percentile(ensemble_result.size());
    auto num_time_points = ensemble_result[0].get_num_time_points();
    auto num_elements    = static_cast<size_t>(mio::mpm::paper::InfectionState::Count);
    for (size_t run = 0; run < ensemble_result.size(); ++run) {
        for (size_t region = 0; region < num_regions; ++region) {
            auto ts = mio::TimeSeries<double>::zero(num_time_points, num_elements);
            for (Eigen::Index time = 0; time < num_time_points; time++) {
                ts.get_time(time) = ensemble_result[run].get_time(time);
                for (Eigen::Index elem = 0; elem < num_elements; elem++) {
                    ts.get_value(time)[elem] = ensemble_result[run].get_value(time)[region * num_elements + elem];
                }
            }
            ensemble_percentile[run].push_back(ts);
        }
    }
    return ensemble_percentile;
}

void percentile_output_to_file(std::vector<mio::TimeSeries<double>>& percentile_output, std::string filename)
{
    auto ts = mio::TimeSeries<double>::zero(percentile_output[0].get_num_time_points(),
                                            percentile_output.size() *
                                                static_cast<size_t>(mio::mpm::paper::InfectionState::Count));
    for (Eigen::Index time = 0; time < percentile_output[0].get_num_time_points(); time++) {
        ts.get_time(time) = percentile_output[0].get_time(time);
        for (size_t region = 0; region < percentile_output.size(); ++region) {
            for (Eigen::Index elem = 0; elem < percentile_output[region].get_num_elements(); elem++) {
                ts.get_value(time)[region * static_cast<size_t>(mio::mpm::paper::InfectionState::Count) + elem] =
                    percentile_output[region].get_value(time)[elem];
            }
        }
    }

    auto file = fopen(filename.c_str(), "w");
    mio::mpm::print_to_file(file, ts, {});
    fclose(file);
}

mio::TimeSeries<double> add_time_series(mio::TimeSeries<double>& t1, mio::TimeSeries<double>& t2)
{
    assert(t1.get_num_time_points() == t2.get_num_time_points());
    mio::TimeSeries<double> added_time_series(t1.get_num_elements());
    auto num_points = static_cast<size_t>(t1.get_num_time_points());
    for (size_t t = 0; t < num_points; ++t) {
        added_time_series.add_time_point(t2.get_time(t), t1.get_value(t) + t2.get_value(t));
    }
    return added_time_series;
}

template <class Model>
void run(Model model, size_t num_runs, double tmax, double dt, size_t num_regions, bool save_percentiles,
         std::string result_prefix)
{
    using Status = mio::mpm::paper::InfectionState;
    std::vector<mio::TimeSeries<double>> ensemble_results(
        num_runs, mio::TimeSeries<double>::zero(tmax, num_regions * static_cast<size_t>(Status::Count)));
    TIME_TYPE total_sim_time = TIME_NOW;
    for (int run = 0; run < num_runs; ++run) {
        std::cout << "run: " << run << "\n";
        auto sim = mio::Simulation<Model>(model, 0.0, dt);
        sim.advance(tmax);
        auto& run_result = sim.get_result();
        // for (size_t i = 0; i < num_regions; ++i) {
        //     for (size_t j = 0; j < num_regions; ++j) {
        //         if (i != j) {
        //             double num_transitions = 0.0;
        //             for (size_t s = 0; s < static_cast<size_t>(Status::Count); ++s) {
        //                 num_transitions += sim.get_model().number_transitions(
        //                     {Status(s), mio::mpm::Region(i), mio::mpm::Region(j), 0.0});
        //             }
        //             std::cout << i << " -> " << j << ": " << num_transitions / tmax << "\n";
        //         }
        //     }
        // }
        ensemble_results[run] = mio::interpolate_simulation_result(run_result);
    }
    restart_timer(total_sim_time, "Time for simulation")
    { // add all results
        mio::TimeSeries<double> mean_time_series =
            std::accumulate(ensemble_results.begin(), ensemble_results.end(),
                            mio::TimeSeries<double>::zero(ensemble_results[0].get_num_time_points(),
                                                          num_regions * static_cast<size_t>(Status::Count)),
                            add_time_series);
        //calculate average
        for (size_t t = 0; t < static_cast<size_t>(mean_time_series.get_num_time_points()); ++t) {
            mean_time_series.get_value(t) *= 1.0 / num_runs;
        }

        std::string dir = mio::base_dir() + "cpp/outputs/" + result_prefix;

        //save mean timeseries
        FILE* file = fopen((dir + "_output_mean.txt").c_str(), "w");
        mio::mpm::print_to_file(file, mean_time_series, {});
        fclose(file);

        if (save_percentiles) {

            auto ensemble_percentile = get_format_for_percentile_output(ensemble_results, num_regions);

            //save percentile output
            auto ensemble_result_p05 = mio::ensemble_percentile(ensemble_percentile, 0.05);
            auto ensemble_result_p25 = mio::ensemble_percentile(ensemble_percentile, 0.25);
            auto ensemble_result_p50 = mio::ensemble_percentile(ensemble_percentile, 0.50);
            auto ensemble_result_p75 = mio::ensemble_percentile(ensemble_percentile, 0.75);
            auto ensemble_result_p95 = mio::ensemble_percentile(ensemble_percentile, 0.95);

            percentile_output_to_file(ensemble_result_p05, dir + "_output_p05.txt");
            percentile_output_to_file(ensemble_result_p25, dir + "_output_p25.txt");
            percentile_output_to_file(ensemble_result_p50, dir + "_output_p50.txt");
            percentile_output_to_file(ensemble_result_p75, dir + "_output_p75.txt");
            percentile_output_to_file(ensemble_result_p95, dir + "_output_p95.txt");
        }
    }
}

int main()
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status = mio::mpm::paper::InfectionState;
    using Region = mio::mpm::Region;
    using ABM    = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    using PDMM   = mio::mpm::PDMModel<8, Status>;

    //parameters for model setup
    // mio::Date start_date             = mio::Date(2021, 3, 1);
    // double t_Exposed                 = 4;
    // double t_Carrier                 = 2.67;
    // double t_Infected                = 5.03;
    // double mu_C_R                    = 0.29;
    // double transmission_rate         = 1.0;
    // double mu_I_D                    = 0.00476;
    // double scaling_factor_trans_rate = 1.0;
    // double contact_radius            = 50;
    // double persons_per_agent         = 300;
    // double tmax                      = 30;
    // double dt                        = 0.1;

    // std::vector<int> regions        = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    // std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};

    // Eigen::MatrixXi metaregions;
    // {
    //     const auto fname = mio::base_dir() + "metagermany.pgm";
    //     std::ifstream ifile(fname);
    //     if (!ifile.is_open()) {
    //         mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
    //         std::abort();
    //     }
    //     else {
    //         metaregions = mio::mpm::read_pgm_raw(ifile).first;
    //         ifile.close();
    //     }
    // }

    // size_t num_regions = metaregions.maxCoeff();

    // WeightedGradient wg(mio::base_dir() + "potentially_germany_grad.json", mio::base_dir() + "boundary_ids.pgm");
    // const std::vector<double> weights(14, 500);
    // const std::vector<double> sigmas(metaregions.maxCoeff(), 10);
    // wg.apply_weights(weights);

    // const std::vector<int> county_ids   = {233, 228, 242, 223, 238, 232, 231, 229};
    // Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();

    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(metaregions.maxCoeff(),
    //                                                                                        metaregions.maxCoeff());
    // commute_weights.setZero();
    // for (int i = 0; i < num_regions; i++) {
    //     for (int j = 0; j < num_regions; j++) {
    //         if (i != j) {
    //             commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
    //         }
    //     }
    //     commute_weights(i, i) = populations[i] - commute_weights.row(i).sum();
    // }

    // std::map<std::tuple<Region, Region>, double> transition_factors{
    //     {{Region(0), Region(1)}, 0.000958613}, {{Region(0), Region(2)}, 0.00172736},
    //     {{Region(0), Region(3)}, 0.0136988},   {{Region(0), Region(4)}, 0.00261568},
    //     {{Region(0), Region(5)}, 0.000317227}, {{Region(0), Region(6)}, 0.000100373},
    //     {{Region(0), Region(7)}, 0.00012256},  {{Region(1), Region(0)}, 0.000825387},
    //     {{Region(1), Region(2)}, 0.00023648},  {{Region(1), Region(3)}, 0.0112213},
    //     {{Region(1), Region(4)}, 0.00202101},  {{Region(1), Region(5)}, 0.00062912},
    //     {{Region(1), Region(6)}, 0.000201067}, {{Region(1), Region(7)}, 0.000146773},
    //     {{Region(2), Region(0)}, 0.000712533}, {{Region(2), Region(1)}, 0.000102613},
    //     {{Region(2), Region(3)}, 0.00675979},  {{Region(2), Region(4)}, 0.00160171},
    //     {{Region(2), Region(5)}, 0.000175467}, {{Region(2), Region(6)}, 0.00010336},
    //     {{Region(2), Region(7)}, 6.21867e-05}, {{Region(3), Region(0)}, 0.00329632},
    //     {{Region(3), Region(1)}, 0.00322347},  {{Region(3), Region(2)}, 0.00412565},
    //     {{Region(3), Region(4)}, 0.0332566},   {{Region(3), Region(5)}, 0.00462197},
    //     {{Region(3), Region(6)}, 0.00659424},  {{Region(3), Region(7)}, 0.00255147},
    //     {{Region(4), Region(0)}, 0.000388373}, {{Region(4), Region(1)}, 0.000406827},
    //     {{Region(4), Region(2)}, 0.000721387}, {{Region(4), Region(3)}, 0.027394},
    //     {{Region(4), Region(5)}, 0.00127328},  {{Region(4), Region(6)}, 0.00068224},
    //     {{Region(4), Region(7)}, 0.00104491},  {{Region(5), Region(0)}, 0.00013728},
    //     {{Region(5), Region(1)}, 0.000475627}, {{Region(5), Region(2)}, 0.00010688},
    //     {{Region(5), Region(3)}, 0.00754293},  {{Region(5), Region(4)}, 0.0034704},
    //     {{Region(5), Region(6)}, 0.00210027},  {{Region(5), Region(7)}, 0.000226667},
    //     {{Region(6), Region(0)}, 7.264e-05},   {{Region(6), Region(1)}, 0.0001424},
    //     {{Region(6), Region(2)}, 9.55733e-05}, {{Region(6), Region(3)}, 0.00921109},
    //     {{Region(6), Region(4)}, 0.0025216},   {{Region(6), Region(5)}, 0.00266944},
    //     {{Region(6), Region(7)}, 0.00156053},  {{Region(7), Region(0)}, 7.81867e-05},
    //     {{Region(7), Region(1)}, 0.0001024},   {{Region(7), Region(2)}, 8.256e-05},
    //     {{Region(7), Region(3)}, 0.00833152},  {{Region(7), Region(4)}, 0.00393717},
    //     {{Region(7), Region(5)}, 0.000354987}, {{Region(7), Region(6)}, 0.00055456}};

    // //create model setup
    // mio::mpm::paper::ModelSetup<ABM::Agent> setup(
    //     t_Exposed, t_Carrier, t_Infected, transmission_rate, mu_C_R, mu_I_D, start_date, regions, populations,
    //     persons_per_agent, metaregions, tmax, dt, commute_weights, wg, sigmas, contact_radius, transition_factors, 10.0);
    size_t num_runs = 1;
    mio::mpm::paper::ModelSetup<ABM::Agent> setup;

    ABM abm   = setup.create_abm<ABM>();
    PDMM pdmm = setup.create_pdmm<PDMM>();

    // std::cout << "Transitions real per day\n";
    // for (size_t i = 0; i < setup.metaregions.maxCoeff(); ++i) {
    //     for (size_t j = 0; j < setup.metaregions.maxCoeff(); ++j) {
    //         if (i != j) {
    //             std::cout << i << " -> " << j << ": "
    //                       << (setup.commute_weights(i, j) + setup.commute_weights(j, i)) / setup.persons_per_agent
    //                       << "\n";
    //         }
    //     }
    // }

    // std::cout << "Commuters real per day\n";
    // for (size_t i = 0; i < setup.metaregions.maxCoeff(); ++i) {
    //     for (size_t j = 0; j < setup.metaregions.maxCoeff(); ++j) {
    //         if (i != j) {
    //             std::cout << i << " -> " << j << ": " << (setup.commute_weights(i, j)) / setup.persons_per_agent
    //                       << "\n";
    //         }
    //     }
    // }

    auto& transition_rates = pdmm.parameters.get<mio::mpm::TransitionRates<Status>>();
    for (auto& rate : transition_rates) {
        rate.factor = (setup.commute_weights(static_cast<size_t>(rate.from), static_cast<size_t>(rate.to)) +
                       setup.commute_weights(static_cast<size_t>(rate.to), static_cast<size_t>(rate.from))) /
                      setup.populations[static_cast<size_t>(rate.from)];
    }

    //std::cout << (setup.commute_weights.array() / setup.persons_per_agent).matrix() << "\n";
    std::cout << "num_agents_pdmm: " << pdmm.populations.get_total() << std::endl;
    std::cout << "num_agents_abm: " << abm.populations.size() << std::endl;

    run(abm, num_runs, setup.tmax, setup.dt, setup.metaregions.maxCoeff(), true, "ABM");
    run(pdmm, num_runs, setup.tmax, setup.dt, setup.metaregions.maxCoeff(), true, "PDMM");

    return 0;
}