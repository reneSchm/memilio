#ifndef MIO_MPM_ANALYZE_RESULT_H_
#define MIO_MPM_ANALYZE_RESULT_H_

#include "hybrid_paper/library/infection_state.h"
#include "memilio/math/eigen.h"
#include "memilio/utils/time_series.h"
#include "memilio/data/analyze_result.h"
#include "mpm/utility.h"

namespace mio
{

namespace mpm
{

TimeSeries<double> accumulate_flows(TimeSeries<double> flow_results)
{
    auto interpolated_flows = interpolate_simulation_result(flow_results);
    TimeSeries<double> acc_flow_ts(flow_results.get_num_elements());
    size_t t     = 0;
    size_t t_int = 0;
    for (; t_int < interpolated_flows.get_num_time_points(); ++t_int) {
        Eigen::VectorXd acc_flows = Eigen::VectorXd::Zero(interpolated_flows.get_num_elements());
        while (flow_results.get_time(t) < interpolated_flows.get_time(t_int)) {
            acc_flows += flow_results.get_value(t);
            ++t;
        }
        acc_flows += interpolated_flows.get_value(t_int);
        if (t_int > 0) {
            acc_flows -= interpolated_flows.get_value(t_int - 1);
        }
        acc_flow_ts.add_time_point(interpolated_flows.get_time(t_int), acc_flows);
        if (flow_results.get_time(t) == interpolated_flows.get_time(t_int)) {
            ++t;
        }
    }
    return acc_flow_ts;
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
    if (file == NULL) {
        mio::log(mio::LogLevel::critical, "Could not open file {}", filename);
    }
    else {
        print_to_file(file, ts, {});
        fclose(file);
    }
}

} // namespace mpm
} // namespace mio

#endif // MIO_MPM_ANALYZE_RESULT_H_