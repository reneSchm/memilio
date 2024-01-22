#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/model_setup.h"
#include "hybrid_paper/library/analyze_result.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "mpm/abm.h"
#include "mpm/pdmm.h"

#include "memilio/data/analyze_result.h"
#include "memilio/utils/random_number_generator.h"

#include <string>

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

    mio::mpm::paper::ModelSetup<ABM::Agent> setup;

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
            // tr->factor = (setup.commute_weights(static_cast<size_t>(tr->from), static_cast<size_t>(tr->to)) +
            //               setup.commute_weights(static_cast<size_t>(tr->to), static_cast<size_t>(tr->from))) /
            //              setup.populations[static_cast<size_t>(tr->from)];
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
        num_runs, mio::TimeSeries<double>::zero(setup.tmax + 1, num_regions * static_cast<size_t>(Status::Count)));
    std::vector<mio::TimeSeries<double>> ensemble_results_flows(
        num_runs, mio::TimeSeries<double>::zero(setup.tmax, num_regions * mio::mpm::paper::flow_indices.size()));

    TIME_TYPE total_sim_time = TIME_NOW;

    // set how many commuters enter the focus region each day
    Eigen::VectorXd posteriori_commute_weight = setup.k_provider.metaregion_commute_weights.col(focus_region);
    posteriori_commute_weight[focus_region]   = 0;

    for (size_t run = 0; run < num_runs; ++run) {
        std::cout << "run: " << run << "\n";
        auto simABM  = mio::Simulation<ABM>(abm, 0.0, setup.dt);
        auto simPDMM = mio::Simulation<PDMM>(pdmm, 0.0, setup.dt);

        for (double t = -setup.dt;; t = std::min(t + setup.dt, setup.tmax)) {
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
                    auto index = simPDMM.get_model().populations.get_flat_index({Region(focus_region), (Status)i});
                    simABM.get_result().get_last_value()[index] += 1;
                    auto& agents = pop[index];
                    for (; agents > 0; agents -= 1) {
                        const double daytime = t - std::floor(t);
                        if (daytime < 13. / 24.) {
                            const size_t commuting_origin =
                                mio::DiscreteDistribution<size_t>::get_instance()(posteriori_commute_weight);
                            const double t_return =
                                std::floor(t) + mio::ParameterDistributionNormal(13.0 / 24.0, 23.0 / 24.0, 18.0 / 24.0)
                                                    .get_rand_sample();
                            simABM.get_model().populations.push_back(
                                {setup.k_provider.metaregion_sampler(focus_region), (Status)i, focus_region, true,
                                 setup.k_provider.metaregion_sampler(commuting_origin), t_return, 0});
                        }
                        else {
                            simABM.get_model().populations.push_back(
                                {setup.k_provider.metaregion_sampler(focus_region), (Status)i, focus_region, false});
                        }
                    }
                }
            }
            if (t >= setup.tmax)
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
        // std::cout << "transitions ABM\n";
        // for (size_t i = 0; i < num_regions; ++i) {
        //     for (size_t j = 0; j < num_regions; ++j) {
        //         if (i != j) {
        //             double num_transitions = 0.0;
        //             for (size_t s = 0; s < static_cast<size_t>(Status::Count); ++s) {
        //                 num_transitions += simABM.get_model().number_transitions(
        //                     {Status(s), mio::mpm::Region(i), mio::mpm::Region(j), 0.0});
        //             }
        //             std::cout << i << " -> " << j << ": " << num_transitions / setup.tmax << "\n";
        //         }
        //     }
        // }
        // std::cout << "transitions PDMM\n";
        // for (size_t i = 0; i < num_regions; ++i) {
        //     for (size_t j = 0; j < num_regions; ++j) {
        //         if (i != j) {
        //             double num_transitions = 0.0;
        //             for (size_t s = 0; s < static_cast<size_t>(Status::Count); ++s) {
        //                 num_transitions += simPDMM.get_model().number_transitions(
        //                     {Status(s), mio::mpm::Region(i), mio::mpm::Region(j), 0.0});
        //             }
        //             std::cout << i << " -> " << j << ": " << num_transitions / setup.tmax << "\n";
        //         }
        //     }
        // }
    }
    restart_timer(total_sim_time, "Time for simulation");

    mio::TimeSeries<double> mean_time_series_comp =
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