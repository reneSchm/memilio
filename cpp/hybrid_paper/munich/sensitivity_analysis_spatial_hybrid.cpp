#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/munich/sensitivity_analysis_setup_munich.h"
#include "memilio/compartments/simulation.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "hybrid_paper/library/sensitivity_analysis.h"
#include "hybrid_paper/munich/munich_setup.h"
#include "memilio/utils/logging.h"
#include "sensitivity_analysis_setup_munich.h"
#include <algorithm>
#include <cstddef>
#include <omp.h>
#include <vector>

void set_up_models(mio::mpm::ABM<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>>& abm,
                   mio::mpm::PDMModel<8, mio::mpm::paper::InfectionState>& pdmm)
{
    pdmm.populations.array().setZero();
    //delete PDMM transition rates in focus region
    auto& transition_rates = pdmm.parameters.get<mio::mpm::TransitionRates<mio::mpm::paper::InfectionState>>();
    for (auto tr = transition_rates.begin(); tr < transition_rates.end();) {
        if (tr->from == mio::mpm::Region(3)) {
            tr = transition_rates.erase(tr);
        }
        else {
            ++tr;
        }
    }
    //delete PDMM adoption rates in focus region
    auto& adoption_rates = pdmm.parameters.get<mio::mpm::AdoptionRates<mio::mpm::paper::InfectionState>>();
    for (auto ar = adoption_rates.begin(); ar < adoption_rates.end(); ++ar) {
        if (ar->region == mio::mpm::Region(3)) {
            ar->factor = 0;
        }
    }

    //delete ABM adoption rates in non-focus regions
    auto& abm_adoption_rates = abm.get_adoption_rates();
    for (auto& ar : abm_adoption_rates) {
        if (std::get<0>(ar.first) != mio::mpm::Region(3)) {
            ar.second.factor = 0;
        }
    }
}

std::vector<double> simulate_hybridization(
    mio::mpm::ABM<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>>& abm,
    mio::mpm::PDMModel<8, mio::mpm::paper::InfectionState>& pdmm,
    const mio::mpm::paper::MunichSetup<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>::Agent>& setup,
    size_t num_runs_per_output)
{
    using Status              = mio::mpm::paper::InfectionState;
    using ABM                 = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    using PDMM                = mio::mpm::PDMModel<8, Status>;
    const size_t focus_region = 3;
    std::vector<mio::TimeSeries<double>> ensemble_results(
        num_runs_per_output,
        mio::TimeSeries<double>::zero(setup.tmax, setup.num_regions * static_cast<size_t>(Status::Count)));
    std::vector<double> timing(num_runs_per_output);

    // set how many commuters enter the focus region each day
    Eigen::VectorXd posteriori_commute_weight = setup.commute_weights.col(focus_region);
    posteriori_commute_weight[focus_region]   = 0;

#pragma omp barrier
#pragma omp parallel for
    for (size_t run = 0; run < num_runs_per_output; ++run) {
        mio::TimeSeries<double> hybrid_result(setup.num_regions * static_cast<size_t>(Status::Count));
        double t_start = omp_get_wtime();
        auto simABM    = mio::Simulation<ABM>(abm, 0.0, setup.dt);
        setup.redraw_agents_status(simABM);
        auto simPDMM = mio::Simulation<PDMM>(pdmm, 0.0, setup.dt);
        for (double t = 0;; t = std::min(t + setup.dt, setup.tmax)) {
            simABM.advance(t);
            simPDMM.advance(t);
            auto& pop        = simPDMM.get_model().populations;
            auto& agents_abm = simABM.get_model().populations;
            auto PDMM_state  = simPDMM.get_result().get_last_value();
            auto ABM_state   = simABM.get_result().get_last_value();
            { //move agents from abm to pdmm
                auto itr = agents_abm.begin();
                while (itr != agents_abm.end()) {
                    if (itr->region != focus_region) {
                        auto index = pop.get_flat_index({mio::mpm::Region(itr->region), itr->status});
                        ABM_state[index] -= 1;
                        PDMM_state[index] += 1;
                        pop[{mio::mpm::Region(itr->region), itr->status}] += 1;
                        itr = agents_abm.erase(itr);
                    }
                    else {
                        itr++;
                    }
                }
            }
            { //move agents from pdmm to abm
                for (int i = 0; i < (int)Status::Count; i++) {
                    auto index           = pop.get_flat_index({mio::mpm::Region(focus_region), (Status)i});
                    auto& agents_to_move = PDMM_state[index];
                    for (; agents_to_move > 0; agents_to_move -= 1) {
                        ABM_state[index] += 1;
                        pop[{mio::mpm::Region(focus_region), (Status)i}] -= 1;
                        const double daytime = t - std::floor(t);
                        //create new ABM agent
                        if (daytime <= 13. / 24.) //agents is commuting to focus region
                        {
                            const size_t commuting_origin =
                                mio::DiscreteDistribution<size_t>::get_instance()(posteriori_commute_weight);
                            //return time has to be drawn
                            const double t_return = std::floor(t) + mio::ParameterDistributionNormal(
                                                                        13.0 / 24.0 + 1.1 * setup.dt,
                                                                        23.0 / 24.0 - 1.1 * setup.dt, 18.0 / 24.0)
                                                                        .get_rand_sample();
                            agents_abm.push_back({setup.metaregion_sampler(focus_region), (Status)i, focus_region, true,
                                                  setup.metaregion_sampler(commuting_origin), t_return, 0});
                        }
                        else //agent returns to focus region
                        {
                            agents_abm.push_back(
                                {setup.metaregion_sampler(focus_region), (Status)i, focus_region, false});
                        }
                    }
                }
            }
            hybrid_result.add_time_point(t, PDMM_state + ABM_state);
            if (t >= setup.tmax) {
                break;
            }
        }
        double t_end = omp_get_wtime();
        //interpolate result
        ensemble_results[run] = mio::interpolate_simulation_result(hybrid_result);
        timing[run]           = t_end - t_start;
    }
    //claculate mean result and mean time
    mio::TimeSeries<double> mean_time_series =
        std::accumulate(ensemble_results.begin(), ensemble_results.end(),
                        mio::TimeSeries<double>::zero(ensemble_results[0].get_num_time_points(),
                                                      setup.num_regions * static_cast<size_t>(Status::Count)),
                        mio::mpm::paper::add_time_series);
    for (size_t t = 0; t < static_cast<size_t>(mean_time_series.get_num_time_points()); ++t) {
        mean_time_series.get_value(t) *= 1.0 / num_runs_per_output;
    }

    double mean_time = (1.0 / num_runs_per_output) * std::accumulate(timing.begin(), timing.end(), 0.0);
    mio::unused(mean_time);
    return std::vector<double>{norm_num_infected<QuadWellSetup<ABM::Agent>>(mean_time_series, setup.num_regions),
                               max_num_infected<QuadWellSetup<ABM::Agent>>(mean_time_series, setup.num_regions),
                               total_transmissions<QuadWellSetup<ABM::Agent>>(mean_time_series, setup.num_regions),
                               total_deaths<QuadWellSetup<ABM::Agent>>(mean_time_series, setup.num_regions)};
}

void run_sensitivity_analysis_hybrid(SensitivitySetupMunich& sensi_setup, size_t num_runs, size_t num_runs_per_output,
                                     double tmax, double dt, size_t num_agents, std::string result_dir)
{
    using Status     = mio::mpm::paper::InfectionState;
    using ABM        = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    using PDMM       = mio::mpm::PDMModel<8, Status>;
    using ModelSetup = mio::mpm::paper::MunichSetup<ABM::Agent>;

    for (size_t run = 0; run < num_runs; ++run) {
        std::cerr << "start run " << run << "\n";
        std::map<std::string, double> base_values(sensi_setup.base_values);
        std::map<std::string, mio::ParameterDistributionUniform> params(sensi_setup.params);
        //draw base values
        draw_base_values(params, base_values);
        //create setup with base values
        const ModelSetup setup_base = create_model_setup<ModelSetup>(base_values, tmax, dt, num_agents);
        //create models with base parameters
        ABM abm_base   = setup_base.create_abm<ABM>();
        PDMM pdmm_base = setup_base.create_pdmm<PDMM>();
        set_up_models(abm_base, pdmm_base);
        std::vector<double> y_base = simulate_hybridization(abm_base, pdmm_base, setup_base, num_runs_per_output);
        for (auto it = base_values.begin(); it != base_values.end(); ++it) {
            double old_value             = it->second;
            it->second                   = old_value + sensi_setup.deltas.at(it->first);
            const ModelSetup setup_delta = create_model_setup<ModelSetup>(base_values, tmax, dt, num_agents);
            //create models with varied parameters
            ABM abm_delta   = setup_delta.create_abm<ABM>();
            PDMM pdmm_delta = setup_delta.create_pdmm<PDMM>();
            set_up_models(abm_delta, pdmm_delta);
            std::vector<double> y_delta =
                simulate_hybridization(abm_delta, pdmm_delta, setup_delta, num_runs_per_output);
            //save outputs
            for (size_t i = 0; i < y_base.size(); ++i) {
                double diff                                    = y_delta[i] - y_base[i];
                sensi_setup.elem_effects[i].at(it->first)[run] = diff / sensi_setup.deltas.at(it->first);
                sensi_setup.diffs[i].at(it->first)[run]        = diff;
                sensi_setup.rel_effects[i].at(it->first)[run]  = diff / (sensi_setup.deltas.at(it->first) / old_value);
            }
            // reset param value
            it->second = old_value;
        }
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
    const size_t num_regions         = 8;
    const size_t num_runs            = 10;
    const size_t num_runs_per_output = 90;
    const size_t num_agents          = 4000;
    double tmax                      = 150.0;
    double dt                        = 0.1;

    std::string result_dir = mio::base_dir() + "cpp/outputs/sensitivity_analysis/20241011_v1/";

    SensitivitySetupMunich sensi_setup(num_runs, 4);
    run_sensitivity_analysis_hybrid(sensi_setup, num_runs, num_runs_per_output, tmax, dt, num_agents,
                                    result_dir + "Hybrid");

    return 0;
}
