#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/quad_well.h"
#include "memilio/compartments/simulation.h"
#include "memilio/data/analyze_result.h"
#include "memilio/math/adapt_rk.h"
#include "memilio/math/integrator.h"
#include "mpm/abm.h"
#include "mpm/pdmm.h"
#include "memilio/math/eigen.h"
#include "quad_well_setup.h"
#include "memilio/utils/time_series.h"
#include "hybrid_paper/library/ensemble_run.h"

#include <cstddef>
#include <memory>
#include <numeric>
#include <omp.h>
#include <ostream>

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()

void run_hybridization(size_t num_runs, size_t num_agents, bool save_percentiles, std::string result_path,
                       bool save_res)
{
    const size_t num_regions = 4;
    const int focus_region   = 0;
    using Status             = mio::mpm::paper::InfectionState;
    using Region             = mio::mpm::Region;
    using ABM                = mio::mpm::ABM<QuadWellModel<Status>>;
    using PDMM               = mio::mpm::PDMModel<4, Status>;

    double time_mean = 0;

    const QuadWellSetup<ABM::Agent> setup(num_agents);

    ABM abm = setup.create_abm<ABM>();
    abm.set_non_moving_regions({1, 2, 3});
    PDMM pdmm = setup.create_pdmm<PDMM>();
    setup.save_setup(result_path);

    //PDMM gets its populations through first exchange timestep
    pdmm.populations.array().setZero();

    //delete transitions rates in focus region in PDMM
    auto& transition_rates = pdmm.parameters.get<mio::mpm::TransitionRates<Status>>();
    for (auto tr = transition_rates.begin(); tr < transition_rates.end();) {
        if (tr->from == Region(focus_region)) {
            tr = transition_rates.erase(tr);
        }
        else {
            ++tr;
        }
    }

    //delete adoption rates in focus region in PDMM
    auto& adoption_rates = pdmm.parameters.get<mio::mpm::AdoptionRates<Status>>();
    for (auto& ar : adoption_rates) {
        if (ar.region == Region(focus_region)) {
            ar.factor = 0;
        }
    }

    //delete adoption rates in ABM in non-fous regions
    auto& abm_adoption_rates = abm.get_adoption_rates();
    for (auto& ar : abm_adoption_rates) {
        if (std::get<0>(ar.first) != Region(focus_region)) {
            ar.second.factor = 0;
        }
    }

    std::vector<mio::TimeSeries<double>> ensemble_results(
        num_runs, mio::TimeSeries<double>(num_regions * static_cast<size_t>(Status::Count)));

    auto& region_rng = mio::DiscreteDistribution<size_t>::get_instance();
    std::cerr << "num_runs " << num_runs << "\n" << std::flush;
#pragma omp barrier
#pragma omp parallel for
    for (size_t run = 0; run < num_runs; ++run) {
        mio::thread_local_rng().seed({static_cast<uint32_t>(run)});
        std::cerr << "Start run " << run << std::endl << std::flush;
        std::vector<double> region_weights(3);
        double t_start = omp_get_wtime();
        auto simABM    = mio::Simulation<ABM>(abm, 0.0, setup.dt);
        setup.redraw_agents_status(simABM);
        auto simPDMM = mio::Simulation<PDMM>(pdmm, 0.0, setup.dt);
        mio::TimeSeries<double> hybrid_result(num_regions * static_cast<size_t>(Status::Count));
        for (double t = 0;; t = std::min(t + setup.dt, setup.tmax)) {
            simABM.advance(t);
            simPDMM.advance(t);
            auto& pop_PDMM         = simPDMM.get_model().populations;
            auto state_PDMM        = simPDMM.get_result().get_last_value();
            auto state_ABM         = simABM.get_result().get_last_value();
            auto& abm_agents       = simABM.get_model().populations;
            auto& abm_transitions  = simABM.get_model().number_transitions()[0];
            auto& pdmm_transitions = simPDMM.get_model().number_transitions()[0];

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
                    region_weights[2]    = 0.001 / (num_agents / 4.0);
                    region_weights[0] =
                        state_PDMM[pop_PDMM.get_flat_index({Region(1), (Status)s})] / (num_agents / 4.0);
                    region_weights[1] =
                        state_PDMM[pop_PDMM.get_flat_index({Region(2), (Status)s})] / (num_agents / 4.0);
                    for (; agents_to_move > 0; agents_to_move -= 1) {
                        //agents_entering_abm_cnt++;
                        state_ABM[index] += 1;
                        pop_PDMM[{Region(focus_region), (Status)s}] -= 1;
                        size_t source_region = region_rng(region_weights);
                        simABM.get_model().populations.push_back({setup.focus_pos_rng(source_region), (Status)s});
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
#pragma omp critical
        {
            double time = t_end - t_start;
            std::cout << "Run  " << run << ": Time: " << time << std::endl << std::flush;
            time_mean += time;
        }
    }
#pragma omp single
    {
        std::cout << "Mean time: " << time_mean / double(num_runs) << std::endl << std::flush;
        //post processing
        if (save_res) {
            {
                //calculate mean
                mio::TimeSeries<double> mean =
                    std::accumulate(ensemble_results.begin(), ensemble_results.end(),
                                    mio::TimeSeries<double>::zero(ensemble_results[0].get_num_time_points(),
                                                                  ensemble_results[0].get_num_elements()),
                                    mio::mpm::paper::add_time_series);
                for (size_t t = 0; t < static_cast<size_t>(mean.get_num_time_points()); ++t) {
                    mean.get_value(t) *= 1.0 / num_runs;
                }

                FILE* file = fopen((result_path + "output_mean.txt").c_str(), "w");
                mio::mpm::print_to_file(file, mean, {});
                fclose(file);

                if (save_percentiles) {
                    auto ensemble_percentile =
                        mio::mpm::paper::get_format_for_percentile_output(ensemble_results, num_regions);

                    //save percentile output
                    auto ensemble_result_p05 = mio::ensemble_percentile(ensemble_percentile, 0.05);
                    auto ensemble_result_p25 = mio::ensemble_percentile(ensemble_percentile, 0.25);
                    auto ensemble_result_p50 = mio::ensemble_percentile(ensemble_percentile, 0.50);
                    auto ensemble_result_p75 = mio::ensemble_percentile(ensemble_percentile, 0.75);
                    auto ensemble_result_p95 = mio::ensemble_percentile(ensemble_percentile, 0.95);

                    mio::mpm::paper::percentile_output_to_file(ensemble_result_p05, result_path + "output_p05.txt");
                    mio::mpm::paper::percentile_output_to_file(ensemble_result_p25, result_path + "output_p25.txt");
                    mio::mpm::paper::percentile_output_to_file(ensemble_result_p50, result_path + "output_p50.txt");
                    mio::mpm::paper::percentile_output_to_file(ensemble_result_p75, result_path + "output_p75.txt");
                    mio::mpm::paper::percentile_output_to_file(ensemble_result_p95, result_path + "output_p95.txt");
                }
            }
        }
    }
}

int main()
{
    mio::set_log_level(mio::LogLevel::warn);
    size_t num_runs        = 140;
    size_t num_agents      = 8000;
    bool save_res          = true;
    bool save_percentiles  = true;
    std::string result_dir = mio::base_dir() + "cpp/outputs/QuadWell/20240923_v1/Hybrid_";
    run_hybridization(num_runs, num_agents, save_percentiles, result_dir, save_res);
    return 0;
}
