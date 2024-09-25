#include "ensemble_run.h"

namespace mio
{

namespace mpm
{

namespace paper
{
std::vector<std::vector<TimeSeries<double>>>
get_format_for_percentile_output(std::vector<TimeSeries<double>>& ensemble_result, size_t num_regions)
{
    std::vector<std::vector<mio::TimeSeries<double>>> ensemble_percentile(ensemble_result.size());
    auto num_time_points = ensemble_result[0].get_num_time_points();
    auto num_elements    = static_cast<size_t>(InfectionState::Count);
    for (size_t run = 0; run < ensemble_result.size(); ++run) {
        for (size_t region = 0; region < num_regions; ++region) {
            auto ts = mio::TimeSeries<double>::zero(num_time_points, num_elements);
            for (Eigen::Index time = 0; time < num_time_points; time++) {
                ts.get_time(time) = ensemble_result[run].get_time(time);
                for (size_t elem = 0; elem < num_elements; elem++) {
                    ts.get_value(time)[elem] = ensemble_result[run].get_value(time)[region * num_elements + elem];
                }
            }
            ensemble_percentile[run].push_back(ts);
        }
    }
    return ensemble_percentile;
}

void percentile_output_to_file(std::vector<TimeSeries<double>>& percentile_output, std::string filename)
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
    print_to_file(file, ts, {});
    fclose(file);
}

TimeSeries<double> add_time_series(TimeSeries<double>& t1, TimeSeries<double>& t2)
{
    assert(t1.get_num_time_points() == t2.get_num_time_points());
    mio::TimeSeries<double> added_time_series(t1.get_num_elements());
    auto num_points = static_cast<size_t>(t1.get_num_time_points());
    for (size_t t = 0; t < num_points; ++t) {
        added_time_series.add_time_point(t2.get_time(t), t1.get_value(t) + t2.get_value(t));
    }
    return added_time_series;
}

void save_results(std::vector<mio::TimeSeries<double>>& ensemble_results, size_t num_runs, size_t num_regions,
                  bool save_percentiles, std::string result_prefix, std::string result_path)
{
    using Status = InfectionState;
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

} // namespace paper
} // namespace mpm
} // namespace mio
