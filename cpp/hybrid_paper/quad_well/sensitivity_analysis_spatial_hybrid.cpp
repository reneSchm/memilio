#include "hybrid_paper/library/quad_well.h"
#include "hybrid_paper/quad_well/quad_well_setup.h"
#include "memilio/data/analyze_result.h"
#include "memilio/utils/logging.h"
#include "memilio/utils/random_number_generator.h"
#include "memilio/utils/time_series.h"
#include "sensitivity_analysis_setup_qw.h"
#include "hybrid_paper/library/sensitivity_analysis.h"
#include <cmath>
#include <cstddef>
#include <iostream>
#include <map>
#include <ostream>
#include <vector>

void set_up_models(mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>& abm,
                   mio::mpm::PDMModel<4, mio::mpm::paper::InfectionState>& pdmm)
{
    abm.set_non_moving_regions({1, 2, 3});
    pdmm.populations.array().setZero();
    //delete transitions rates in focus region in PDMM
    auto& transition_rates = pdmm.parameters.get<mio::mpm::TransitionRates<mio::mpm::paper::InfectionState>>();
    for (auto tr = transition_rates.begin(); tr < transition_rates.end();) {
        if (tr->from == mio::mpm::Region(0)) {
            tr = transition_rates.erase(tr);
        }
        else {
            ++tr;
        }
    }
    //delete adoption rates in focus region in PDMM
    auto& adoption_rates = pdmm.parameters.get<mio::mpm::AdoptionRates<mio::mpm::paper::InfectionState>>();
    for (auto& ar : adoption_rates) {
        if (ar.region == mio::mpm::Region(0)) {
            ar.factor = 0;
        }
    }
    //delete adoption rates in ABM in non-fous regions
    auto& abm_adoption_rates = abm.get_adoption_rates();
    for (auto& ar : abm_adoption_rates) {
        if (std::get<0>(ar.first) != mio::mpm::Region(0)) {
            ar.second.factor = 0;
        }
    }
}

std::vector<double>
simulate_hybridization(mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>& abm,
                       mio::mpm::PDMModel<4, mio::mpm::paper::InfectionState>& pdmm,
                       const QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>& setup,
                       size_t num_runs)
{
    using Status              = mio::mpm::paper::InfectionState;
    using Region              = mio::mpm::Region;
    using ABM                 = mio::mpm::ABM<QuadWellModel<Status>>;
    using PDMM                = mio::mpm::PDMModel<4, Status>;
    const size_t focus_region = 0;

    std::vector<double> region_weights(3);
    auto& region_rng = mio::DiscreteDistribution<size_t>::get_instance();
    std::vector<mio::TimeSeries<double>> ensemble_results(
        num_runs, mio::TimeSeries<double>::zero(setup.tmax, setup.num_regions * static_cast<size_t>(Status::Count)));
    std::vector<double> timing(num_runs);
#pragma omp barrier
#pragma omp parallel for
    for (size_t run = 0; run < num_runs; ++run) {
        std::cerr << "Start inner run " << run << "\n" << std::flush;
        double t_start = omp_get_wtime();
        auto simABM    = mio::Simulation<ABM>(abm, 0.0, setup.dt);
        setup.redraw_agents_status(simABM);
        auto simPDMM = mio::Simulation<PDMM>(pdmm, 0.0, setup.dt);
        //result time series
        mio::TimeSeries<double> hybrid_result(4 * static_cast<size_t>(Status::Count));
        for (double t = 0;; t = std::min(t + setup.dt, setup.tmax)) {
            simABM.advance(t);
            simPDMM.advance(t);

            auto& abm_agents = simABM.get_model().populations;
            auto& pop_PDMM   = simPDMM.get_model().populations;
            auto state_PDMM  = simPDMM.get_result().get_last_value();
            auto state_ABM   = simABM.get_result().get_last_value();
            { //move agents from abm to pdmm
                auto itr = abm_agents.begin();
                while (itr != abm_agents.end()) {
                    auto pos = itr->position;
                    if (qw::well_index(pos) != focus_region) {
                        auto region  = Region(qw::well_index(pos));
                        size_t index = pop_PDMM.get_flat_index({region, itr->status});
                        pop_PDMM[{region, itr->status}] += 1;
                        state_ABM[index] -= 1;
                        state_PDMM[index] += 1;
                        itr = abm_agents.erase(itr);
                    }
                    else {
                        ++itr;
                    }
                }
            }
            { //move agents from pdmm to abm
                for (size_t s = 0; s < static_cast<size_t>(Status::Count); ++s) {
                    size_t index         = pop_PDMM.get_flat_index({Region(focus_region), (Status)s});
                    auto& agents_to_move = state_PDMM[index];
                    //as in the ABM agents barely move between diagonal regions, the wheight for that is very low
                    region_weights[2] = 0.001 / (setup.num_agents / 4.0);
                    //as we do not know from which region the agent came, we need to sample that
                    region_weights[0] =
                        state_PDMM[pop_PDMM.get_flat_index({Region(1), (Status)s})] / (setup.num_agents / 4.0);
                    region_weights[1] =
                        state_PDMM[pop_PDMM.get_flat_index({Region(2), (Status)s})] / (setup.num_agents / 4.0);
                    for (; agents_to_move > 0; agents_to_move -= 1) {
                        state_ABM[index] += 1;
                        pop_PDMM[{Region(focus_region), (Status)s}] -= 1;
                        size_t source_region = region_rng(region_weights);
                        auto new_pos         = setup.focus_pos_rng(source_region);
                        while (qw::well_index(new_pos) != focus_region) {
                            mio::log_warning("Position has to be resampled. x is {:.5f}, y is {:.5f}", new_pos[0],
                                             new_pos[1]);
                            new_pos = setup.focus_pos_rng(source_region);
                        }
                        simABM.get_model().populations.push_back({new_pos, (Status)s});
                    }
                }
            }
            hybrid_result.add_time_point(t, state_ABM + state_PDMM);
            if (t >= setup.tmax) {
                break;
            }
        }
        double t_end          = omp_get_wtime();
        ensemble_results[run] = mio::interpolate_simulation_result(hybrid_result);
        timing[run]           = t_end - t_start;
        std::cerr << "End inner run " << run << "\n" << std::flush;
    }
    //claculate mean result and mean time
    mio::TimeSeries<double> mean_time_series =
        std::accumulate(ensemble_results.begin(), ensemble_results.end(),
                        mio::TimeSeries<double>::zero(ensemble_results[0].get_num_time_points(),
                                                      setup.num_regions * static_cast<size_t>(Status::Count)),
                        mio::mpm::paper::add_time_series);
    for (size_t t = 0; t < static_cast<size_t>(mean_time_series.get_num_time_points()); ++t) {
        mean_time_series.get_value(t) *= 1.0 / num_runs;
    }

    double mean_time = (1.0 / num_runs) * std::accumulate(timing.begin(), timing.end(), 0.0);
    mio::unused(mean_time);
    return std::vector<double>{norm_num_infected<QuadWellSetup<ABM::Agent>>(mean_time_series, setup.num_regions),
                               max_num_infected<QuadWellSetup<ABM::Agent>>(mean_time_series, setup.num_regions),
                               total_transmissions<QuadWellSetup<ABM::Agent>>(mean_time_series, setup.num_regions),
                               total_deaths<QuadWellSetup<ABM::Agent>>(mean_time_series, setup.num_regions)};
}

void run_sensitivity_analysis_hybrid(SensitivitySetupQW& sensi_setup, size_t num_runs, size_t num_runs_per_output,
                                     size_t num_agents, double tmax, double dt, std::string result_dir)
{
    using Status = mio::mpm::paper::InfectionState;
    using Region = mio::mpm::Region;
    using ABM    = mio::mpm::ABM<QuadWellModel<Status>>;
    using PDMM   = mio::mpm::PDMModel<4, Status>;

    const std::vector<double> sigmas      = {0.4, 0.42, 0.44, 0.45, 0.46, 0.48, 0.5, 0.52, 0.54, 0.55, 0.56, 0.58, 0.6};
    const std::vector<double> trans_rates = {0.00002, 0.00006, 0.00013, 0.00019, 0.0003, 0.00059, 0.00115,
                                             0.0021,  0.0035,  0.0044,  0.0055,  0.0085, 0.0124};
    auto& sigma_rng                       = mio::DiscreteDistribution<size_t>::get_instance();
    // #pragma omp barrier
    // #pragma omp parallel for
    for (size_t run = 0; run < num_runs; ++run) {
        std::cerr << "Start outer run " << run << "\n" << std::flush;
        //draw base values
        std::map<std::string, double> base_values(sensi_setup.base_values);
        std::map<std::string, mio::ParameterDistributionUniform> params(sensi_setup.params);
        draw_base_values(params, base_values);
        //sigma and transition rates have to match in hybrid model,
        // therefore we fix them for the analysis
        size_t sigma_index = sigma_rng(std::vector<double>(sigmas.size() - 1, 1.0));
        if (sigma_index == sigmas.size()) {
            mio::log_error("Sigma index is too big");
        }
        base_values.at("sigma")            = sigmas[sigma_index];
        base_values.at("transition_rates") = trans_rates[sigma_index];
        const QuadWellSetup<ABM::Agent> setup_base =
            create_model_setup<QuadWellSetup<ABM::Agent>>(base_values, tmax, dt, num_agents);

        //create models with base parameters
        ABM abm_base   = setup_base.create_abm<ABM>();
        PDMM pdmm_base = setup_base.create_pdmm<PDMM>();
        set_up_models(abm_base, pdmm_base);
        std::vector<double> y_base = simulate_hybridization(abm_base, pdmm_base, setup_base, num_runs_per_output);
        for (auto it = base_values.begin(); it != base_values.end(); ++it) {
            double old_value = it->second;
            // modify param value by delta
            if (it->first == "sigma" || it->first == "transition_rates") {
                base_values.at("sigma")            = sigmas[sigma_index + 1];
                base_values.at("transition_rates") = trans_rates[sigma_index + 1];
            }
            else {
                it->second = old_value + sensi_setup.deltas.at(it->first);
            }
            const QuadWellSetup<ABM::Agent> setup_delta =
                create_model_setup<QuadWellSetup<ABM::Agent>>(base_values, tmax, dt, num_agents);
            //create models with base parameters
            ABM abm_delta   = setup_delta.create_abm<ABM>();
            PDMM pdmm_delta = setup_delta.create_pdmm<PDMM>();
            set_up_models(abm_delta, pdmm_delta);
            std::vector<double> y_delta =
                simulate_hybridization(abm_delta, pdmm_delta, setup_delta, num_runs_per_output);
            // save elementary effect sample
            for (size_t i = 0; i < y_base.size(); ++i) {
                double diff                                    = y_delta[i] - y_base[i];
                sensi_setup.elem_effects[i].at(it->first)[run] = diff / sensi_setup.deltas.at(it->first);
                sensi_setup.diffs[i].at(it->first)[run]        = diff;
                sensi_setup.rel_effects[i].at(it->first)[run]  = diff / (sensi_setup.deltas.at(it->first) / old_value);
            }
            // reset param value
            it->second = old_value;
        }
        std::cerr << "End outer run " << run << "\n" << std::flush;
    }
#pragma omp single
    {
        std::string result_file_elem_eff    = result_dir + "_elem_effects";
        std::string result_file_diff        = result_dir + "_diff";
        std::string result_file_rel_effects = result_dir + "_rel_effects";
        save_elementary_effects(sensi_setup.elem_effects, result_file_elem_eff, num_runs);
        save_elementary_effects(sensi_setup.diffs, result_file_diff, num_runs);
        save_elementary_effects(sensi_setup.rel_effects, result_file_rel_effects, num_runs);
    }
}

int main()
{
    mio::set_log_level(mio::LogLevel::warn);

    const size_t num_regions         = 4;
    const size_t num_runs            = 1;
    const size_t num_runs_per_output = 56;
    const size_t num_agents          = 8000;
    double tmax                      = 150.0;
    double dt                        = 0.1;

    std::string result_dir = mio::base_dir() + "cpp/outputs/sensitivity_analysis/20240930_v2/";

    SensitivitySetupQW sensi_setup(num_runs, 4);
    run_sensitivity_analysis_hybrid(sensi_setup, num_runs, num_runs_per_output, num_agents, tmax, dt,
                                    result_dir + "Hybrid");

    return 0;
}
