#ifndef ENSEMBLE_RUN_H
#define ENSEMBLE_RUN_H

#include "hybrid_paper/library/quad_well.h"
#include "infection_state.h"
#include "initialization.h"
#include "memilio/utils/time_series.h"
#include "models/mpm/utility.h"
#include "memilio/data/analyze_result.h"

#include <cstddef>
#include <iostream>
#include <omp.h>
#include <ostream>

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

namespace mio
{

namespace mpm
{

namespace paper
{

std::vector<std::vector<TimeSeries<double>>>
get_format_for_percentile_output(std::vector<TimeSeries<double>>& ensemble_result, size_t num_regions);

void percentile_output_to_file(std::vector<TimeSeries<double>>& percentile_output, std::string filename);

TimeSeries<double> add_time_series(TimeSeries<double>& t1, TimeSeries<double>& t2);

template <class Model, class SampleStatusFunction>
void run(Model model, size_t num_runs, double tmax, double dt, size_t num_regions, bool save_percentiles,
         std::string result_prefix, std::string result_path, SampleStatusFunction sample_function)
{
    using Status = InfectionState;
    std::vector<mio::TimeSeries<double>> ensemble_results(
        num_runs, mio::TimeSeries<double>::zero(tmax, num_regions * static_cast<size_t>(Status::Count)));
    double time_mean         = 0;
    TIME_TYPE total_sim_time = TIME_NOW;
#pragma omp barrier
#pragma omp parallel for
    for (int run = 0; run < num_runs; ++run) {
        std::cerr << "Start run " << run << "\n" << std::flush;
        double t_start = omp_get_wtime();
        auto sim       = mio::Simulation<Model>(model, 0.0, dt);
        sample_function(sim);
        sim.advance(tmax);
        double t_end     = omp_get_wtime();
        auto& run_result = sim.get_result();

        ensemble_results[run] = mio::interpolate_simulation_result(run_result);
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
        restart_timer(total_sim_time, "Time for simulation");
        // add all results
        mio::TimeSeries<double> mean_time_series =
            std::accumulate(ensemble_results.begin(), ensemble_results.end(),
                            mio::TimeSeries<double>::zero(ensemble_results[0].get_num_time_points(),
                                                          num_regions * static_cast<size_t>(Status::Count)),
                            add_time_series);
        //calculate average
        for (size_t t = 0; t < static_cast<size_t>(mean_time_series.get_num_time_points()); ++t) {
            mean_time_series.get_value(t) *= 1.0 / num_runs;
        }

        std::string dir = result_path + result_prefix;

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

} // namespace paper
} // namespace mpm
} // namespace mio

#endif //ENSEMBLE_RUN_H