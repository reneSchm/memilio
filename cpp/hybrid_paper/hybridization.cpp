#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/model_setup.h"
#include "hybrid_paper/library/analyze_result.h"
#include "library/analyze_result.h"
#include "library/infection_state.h"
#include "mpm/abm.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "mpm/pdmm.h"
#include "memilio/data/analyze_result.h"
#include <cstddef>
#include <cstdio>
#include <string>
#include <utility>

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

void run_simulation(size_t num_runs, bool save_percentiles, bool save_single_outputs)
{
    using Status             = mio::mpm::paper::InfectionState;
    using Region             = mio::mpm::Region;
    using ABM                = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    using PDMM               = mio::mpm::PDMModel<8, Status>;
    const size_t num_regions = 8;
    const int focus_region   = 3;

    mio::Date start_date             = mio::Date(2021, 3, 1);
    double t_Exposed                 = 3.67652;
    double t_Carrier                 = 2.71414;
    double t_Infected                = 5;
    double mu_C_R                    = 0.1;
    double transmission_rate         = 0.35;
    double mu_I_D                    = 0.004;
    double scaling_factor_trans_rate = 1.0; /// 6.48;
    double contact_radius            = 10;
    double persons_per_agent         = 100;
    double tmax                      = 30;
    double dt                        = 0.1;

    std::vector<int> regions        = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};

    Eigen::MatrixXi metaregions;
    {
        const auto fname = mio::base_dir() + "metagermany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            std::abort();
        }
        else {
            metaregions = mio::mpm::read_pgm_raw(ifile).first;
            ifile.close();
        }
    }

    WeightedGradient wg(mio::base_dir() + "potentially_germany_grad.json", mio::base_dir() + "boundary_ids.pgm");
    const std::vector<double> weights(14, 500);
    const std::vector<double> sigmas(num_regions, 10);
    wg.apply_weights(weights);

    const std::vector<int> county_ids   = {233, 228, 242, 223, 238, 232, 231, 229};
    Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(num_regions, num_regions);
    commute_weights.setZero();
    for (int i = 0; i < num_regions; i++) {
        for (int j = 0; j < num_regions; j++) {
            if (i != j) {
                commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
            }
        }
        commute_weights(i, i) = populations[i] - commute_weights.row(i).sum();
    }

    std::map<std::tuple<Region, Region>, double> transition_factors{
        {{Region(0), Region(1)}, 0.000958613}, {{Region(0), Region(2)}, 0.00172736},
        {{Region(0), Region(3)}, 0.0136988},   {{Region(0), Region(4)}, 0.00261568},
        {{Region(0), Region(5)}, 0.000317227}, {{Region(0), Region(6)}, 0.000100373},
        {{Region(0), Region(7)}, 0.00012256},  {{Region(1), Region(0)}, 0.000825387},
        {{Region(1), Region(2)}, 0.00023648},  {{Region(1), Region(3)}, 0.0112213},
        {{Region(1), Region(4)}, 0.00202101},  {{Region(1), Region(5)}, 0.00062912},
        {{Region(1), Region(6)}, 0.000201067}, {{Region(1), Region(7)}, 0.000146773},
        {{Region(2), Region(0)}, 0.000712533}, {{Region(2), Region(1)}, 0.000102613},
        {{Region(2), Region(3)}, 0.00675979},  {{Region(2), Region(4)}, 0.00160171},
        {{Region(2), Region(5)}, 0.000175467}, {{Region(2), Region(6)}, 0.00010336},
        {{Region(2), Region(7)}, 6.21867e-05}, {{Region(3), Region(0)}, 0.00329632},
        {{Region(3), Region(1)}, 0.00322347},  {{Region(3), Region(2)}, 0.00412565},
        {{Region(3), Region(4)}, 0.0332566},   {{Region(3), Region(5)}, 0.00462197},
        {{Region(3), Region(6)}, 0.00659424},  {{Region(3), Region(7)}, 0.00255147},
        {{Region(4), Region(0)}, 0.000388373}, {{Region(4), Region(1)}, 0.000406827},
        {{Region(4), Region(2)}, 0.000721387}, {{Region(4), Region(3)}, 0.027394},
        {{Region(4), Region(5)}, 0.00127328},  {{Region(4), Region(6)}, 0.00068224},
        {{Region(4), Region(7)}, 0.00104491},  {{Region(5), Region(0)}, 0.00013728},
        {{Region(5), Region(1)}, 0.000475627}, {{Region(5), Region(2)}, 0.00010688},
        {{Region(5), Region(3)}, 0.00754293},  {{Region(5), Region(4)}, 0.0034704},
        {{Region(5), Region(6)}, 0.00210027},  {{Region(5), Region(7)}, 0.000226667},
        {{Region(6), Region(0)}, 7.264e-05},   {{Region(6), Region(1)}, 0.0001424},
        {{Region(6), Region(2)}, 9.55733e-05}, {{Region(6), Region(3)}, 0.00921109},
        {{Region(6), Region(4)}, 0.0025216},   {{Region(6), Region(5)}, 0.00266944},
        {{Region(6), Region(7)}, 0.00156053},  {{Region(7), Region(0)}, 7.81867e-05},
        {{Region(7), Region(1)}, 0.0001024},   {{Region(7), Region(2)}, 8.256e-05},
        {{Region(7), Region(3)}, 0.00833152},  {{Region(7), Region(4)}, 0.00393717},
        {{Region(7), Region(5)}, 0.000354987}, {{Region(7), Region(6)}, 0.00055456}};

    mio::mpm::paper::ModelSetup<ABM::Agent> setup(t_Exposed, t_Carrier, t_Infected, transmission_rate, mu_C_R, mu_I_D,
                                                  start_date, regions, populations, persons_per_agent, metaregions,
                                                  tmax, dt, commute_weights, wg, sigmas, contact_radius,
                                                  transition_factors, 1.0, true);

    ABM abm   = setup.create_abm<ABM>();
    PDMM pdmm = setup.create_pdmm<PDMM>();

    std::cout << "num agents: " << abm.populations.size() << std::endl;

    pdmm.populations.array().setZero();

    auto& transition_rates = pdmm.parameters.get<mio::mpm::TransitionRates<Status>>();
    for (auto tr = transition_rates.begin(); tr < transition_rates.end();) {
        if (tr->from == Region(focus_region)) {
            tr = transition_rates.erase(tr);
        }
        else {
            ++tr;
        }
    }
    auto& adoption_rates = pdmm.parameters.get<mio::mpm::AdoptionRates<Status>>();
    for (auto ar = adoption_rates.begin(); ar < adoption_rates.end(); ++ar) {
        if (ar->region == Region(focus_region)) {
            ar->factor = 0;
        }
    }

    auto& abm_adoption_rates = abm.get_adoption_rates();
    for (auto& ar : abm_adoption_rates) {
        if (std::get<0>(ar.first) != Region(focus_region)) {
            ar.second.factor = 0;
        }
    }

    std::vector<mio::TimeSeries<double>> ensemble_results_comp(
        num_runs, mio::TimeSeries<double>::zero(tmax + 1, num_regions * static_cast<size_t>(Status::Count)));
    std::vector<mio::TimeSeries<double>> ensemble_results_flows(
        num_runs, mio::TimeSeries<double>::zero(tmax, num_regions * mio::mpm::paper::flow_indices.size()));

    TIME_TYPE total_sim_time = TIME_NOW;

    for (size_t run = 0; run < num_runs; ++run) {
        std::cout << "run: " << run << "\n";
        auto simABM  = mio::Simulation<ABM>(abm, 0.0, dt);
        auto simPDMM = mio::Simulation<PDMM>(pdmm, 0.0, dt);

        for (double t = -dt;; t = std::min(t + dt, tmax)) {
            simABM.advance(t);
            simPDMM.advance(t);
            { //move agents from abm to pdmm
                const auto pop     = simPDMM.get_model().populations;
                auto simPDMM_state = simPDMM.get_result().get_last_value();
                auto& agents       = simABM.get_model().populations;
                auto itr           = agents.begin();
                while (itr != agents.end()) {
                    if (itr->region != focus_region) {
                        simABM.get_result().get_last_value()[simPDMM.get_model().populations.get_flat_index(
                            {Region(itr->region), itr->status})] -= 1;
                        simPDMM_state[pop.get_flat_index({Region(itr->region), itr->status})] += 1;
                        itr = agents.erase(itr);
                    }
                    else {
                        itr++;
                    }
                }
            }
            { //move agents from pdmm to abm
                auto pop = simPDMM.get_result().get_last_value();
                for (int i = 0; i < (int)Status::Count; i++) {
                    // auto p = simPDMM_state[pop.get_flat_index({focus_region, (InfectionState)i})];
                    auto index   = simPDMM.get_model().populations.get_flat_index({Region(focus_region), (Status)i});
                    auto& agents = pop[index];
                    for (; agents > 0; agents -= 1) {
                        simABM.get_model().populations.push_back(
                            {setup.k_provider.metaregion_sampler(focus_region), (Status)i, focus_region});
                        simABM.get_result().get_last_value()[index] += 1;
                    }
                }
            }
            if (t >= tmax)
                break;
        }

        mio::TimeSeries<double> hybrid_result(num_regions * static_cast<size_t>(Status::Count));
        mio::TimeSeries<double> hybrid_flow_result(num_regions * mio::mpm::paper::flow_indices.size());

        //save hybrid result
        auto interpolated_abm_comps  = mio::interpolate_simulation_result(simABM.get_result());
        auto interpolated_pdmm_comps = mio::interpolate_simulation_result(simPDMM.get_result());

        for (size_t t = 0; t < interpolated_pdmm_comps.get_num_time_points(); ++t) {
            hybrid_result.add_time_point(interpolated_pdmm_comps.get_time(t),
                                         interpolated_pdmm_comps.get_value(t) + interpolated_abm_comps.get_value(t));
        }

        auto accumulated_abm_flows  = mio::mpm::accumulate_flows(simABM.get_model().get_flow_result());
        auto accumulated_pdmm_flows = mio::mpm::accumulate_flows(*(simPDMM.get_model().all_flows));

        for (size_t t = 0; t < accumulated_abm_flows.get_num_time_points(); ++t) {
            hybrid_flow_result.add_time_point(accumulated_abm_flows.get_time(t),
                                              accumulated_abm_flows.get_value(t) + accumulated_pdmm_flows.get_value(t));
        }
        ensemble_results_comp[run]  = hybrid_result;
        ensemble_results_flows[run] = hybrid_flow_result;

        if (save_single_outputs) {
            // auto file = fopen((std::to_string(run)+"_comp_ouput_ABM.txt").c_str(), "w");
            // mio::mpm::print_to_file(file, mio::interpolate_simulation_result(simABM.get_result()), {});
            // fclose(file);

            // auto file1 = fopen((std::to_string(run)+"comp_output_PDMM.txt").c_str(), "w");
            // mio::mpm::print_to_file(file1, mio::interpolate_simulation_result(simPDMM.get_result()), {});
            // fclose(file1);

            // auto file2 = fopen((std::to_string(run)+"flow_output_ABM.txt").c_str(), "w");
            // mio::mpm::print_to_file(file2, simABM.get_model().get_flow_result(), {});
            // fclose(file2);

            // auto file3 = fopen((std::to_string(run)+"flow_output_PDMM.txt").c_str(), "w");
            // mio::mpm::print_to_file(file3, *(simPDMM.get_model().all_flows), {});
            // fclose(file3);

            auto file4 = fopen((std::to_string(run) + "comp_output_Hybrid.txt").c_str(), "w");
            mio::mpm::print_to_file(file4, hybrid_result, {});
            fclose(file4);

            auto file5 = fopen((std::to_string(run) + "flow_output_Hybrid.txt").c_str(), "w");
            mio::mpm::print_to_file(file5, hybrid_flow_result, {});
            fclose(file5);
        }
    }
    restart_timer(total_sim_time, "Time for simulation")

        mio::TimeSeries<double>
            mean_time_series_comp =
                std::accumulate(ensemble_results_comp.begin(), ensemble_results_comp.end(),
                                mio::TimeSeries<double>::zero(ensemble_results_comp[0].get_num_time_points(),
                                                              ensemble_results_comp[0].get_num_elements()),
                                mio::mpm::add_time_series);

    mio::TimeSeries<double> mean_time_series_flows =
        std::accumulate(ensemble_results_flows.begin(), ensemble_results_flows.end(),
                        mio::TimeSeries<double>::zero(ensemble_results_flows[0].get_num_time_points(),
                                                      ensemble_results_flows[0].get_num_elements()),
                        mio::mpm::add_time_series);
    //calculate average
    for (size_t t = 0; t < static_cast<size_t>(mean_time_series_comp.get_num_time_points()); ++t) {
        mean_time_series_comp.get_value(t) *= 1.0 / num_runs;
    }

    for (size_t t = 0; t < static_cast<size_t>(mean_time_series_flows.get_num_time_points()); ++t) {
        mean_time_series_flows.get_value(t) *= 1.0 / num_runs;
    }

    std::string dir = mio::base_dir() + "cpp/outputs/";

    //save mean timeseries
    FILE* file_mean_comp = fopen((dir + "comps_output_mean.txt").c_str(), "w");
    mio::mpm::print_to_file(file_mean_comp, mean_time_series_comp, {});
    fclose(file_mean_comp);

    FILE* file_mean_flows = fopen((dir + "flows_output_mean.txt").c_str(), "w");
    mio::mpm::print_to_file(file_mean_flows, mean_time_series_flows, {});
    fclose(file_mean_flows);

    if (save_percentiles) {

        auto ensemble_percentile_comps = mio::mpm::get_format_for_percentile_output(ensemble_results_comp, num_regions);
        auto ensemble_percentile_flows =
            mio::mpm::get_format_for_percentile_output(ensemble_results_flows, num_regions);

        //save percentile output
        auto ensemble_result_comps_p05 = mio::ensemble_percentile(ensemble_percentile_comps, 0.05);
        auto ensemble_result_comps_p25 = mio::ensemble_percentile(ensemble_percentile_comps, 0.25);
        auto ensemble_result_comps_p50 = mio::ensemble_percentile(ensemble_percentile_comps, 0.50);
        auto ensemble_result_comps_p75 = mio::ensemble_percentile(ensemble_percentile_comps, 0.75);
        auto ensemble_result_comps_p95 = mio::ensemble_percentile(ensemble_percentile_comps, 0.95);

        auto ensemble_result_flows_p05 = mio::ensemble_percentile(ensemble_percentile_flows, 0.05);
        auto ensemble_result_flows_p25 = mio::ensemble_percentile(ensemble_percentile_flows, 0.25);
        auto ensemble_result_flows_p50 = mio::ensemble_percentile(ensemble_percentile_flows, 0.50);
        auto ensemble_result_flows_p75 = mio::ensemble_percentile(ensemble_percentile_flows, 0.75);
        auto ensemble_result_flows_p95 = mio::ensemble_percentile(ensemble_percentile_flows, 0.95);

        mio::mpm::percentile_output_to_file(ensemble_result_comps_p05, dir + "comps_output_p05.txt");
        mio::mpm::percentile_output_to_file(ensemble_result_comps_p25, dir + "comps_output_p25.txt");
        mio::mpm::percentile_output_to_file(ensemble_result_comps_p50, dir + "comps_output_p50.txt");
        mio::mpm::percentile_output_to_file(ensemble_result_comps_p75, dir + "comps_output_p75.txt");
        mio::mpm::percentile_output_to_file(ensemble_result_comps_p95, dir + "comps_output_p95.txt");

        mio::mpm::percentile_output_to_file(ensemble_result_flows_p05, dir + "flows_output_p05.txt");
        mio::mpm::percentile_output_to_file(ensemble_result_flows_p25, dir + "flows_output_p25.txt");
        mio::mpm::percentile_output_to_file(ensemble_result_flows_p50, dir + "flows_output_p50.txt");
        mio::mpm::percentile_output_to_file(ensemble_result_flows_p75, dir + "flows_output_p75.txt");
        mio::mpm::percentile_output_to_file(ensemble_result_flows_p95, dir + "flows_output_p95.txt");
    }
}

void save_new_infections(mio::Date start_date, size_t num_days)
{
    const std::vector<int> regions = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    std::vector<mio::ConfirmedCasesDataEntry> confirmed_new_infections =
        mio::read_confirmed_cases_data(mio::base_dir() + "/data/Germany/new_infections_all_county.json").value();
    mio::TimeSeries<double> new_infections_ts(regions.size());
    mio::Date date = mio::offset_date_by_days(start_date, 1);
    for (size_t d = 0; d < num_days; ++d) {
        auto confirmed_per_region = get_cases_at_date(confirmed_new_infections, regions, date);
        Eigen::VectorXd confirmed_vector(regions.size());
        for (size_t region = 0; region < regions.size(); ++region) {
            confirmed_vector[region] = confirmed_per_region.at(regions[region]);
        }
        new_infections_ts.add_time_point(d + 1, confirmed_vector);
        date = mio::offset_date_by_days(date, 1);
    }

    std::string dir = mio::base_dir() + "cpp/outputs/";

    //save mean timeseries
    FILE* file = fopen((dir + "new_infections.txt").c_str(), "w");
    mio::mpm::print_to_file(file, new_infections_ts, {});
    fclose(file);
}

int main(int argc, char** argv)
{
    mio::set_log_level(mio::LogLevel::warn);
    run_simulation(10, true, false);
    save_new_infections(mio::Date(2021, 3, 1), 30);
    return 0;
}