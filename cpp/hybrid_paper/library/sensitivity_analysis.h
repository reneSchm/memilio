#ifndef SENSITIVITY_ANALYSIS_H
#define SENSITIVITY_ANALYSIS_H

#include "hybrid_paper/quad_well/quad_well_setup.h"
#include "hybrid_paper/library/ensemble_run.h"
#include "memilio/utils/compiler_diagnostics.h"
#include "memilio/utils/time_series.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "memilio/data/analyze_result.h"
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <map>
#include <string>
#include <vector>

template <class ModelSetup>
auto create_model_setup(std::map<std::string, double>& /*params*/, double /*tmax*/, double /*dt*/,
                        size_t /*num_agents*/) -> ModelSetup
{
    return ModelSetup();
}

template <>
QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>
create_model_setup<QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>>(
    std::map<std::string, double>& params, double tmax, double dt, size_t num_agents);

template <class ModelSetup, class Model>
Model create_model(const ModelSetup& model_setup)
{
    return model_setup.create_model();
}

template <>
mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>
create_model<QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>,
             mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>>(
    const QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>& model_setup);

template <>
mio::mpm::PDMModel<4, mio::mpm::paper::InfectionState>
create_model<QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>,
             mio::mpm::PDMModel<4, mio::mpm::paper::InfectionState>>(
    const QuadWellSetup<mio::mpm::ABM<QuadWellModel<mio::mpm::paper::InfectionState>>::Agent>& model_setup);

void save_elementary_effects(std::vector<std::map<std::string, std::vector<double>>>& elem_effects,
                             std::string result_file, size_t num_runs);
// template <class SensiSetup>
// void draw_base_values(const SensiSetup& sensi_setup, std::map<std::string, double>& base_values)
// {
//     for (auto it = base_values.begin(); it != base_values.end(); ++it) {
//         it->second = sensi_setup.params.at(it->first).get_rand_sample();
//     }
// }

void draw_base_values(std::map<std::string, mio::ParameterDistributionUniform>& params,
                      std::map<std::string, double>& base_values);

template <class SensiSetup, class Model, class ModelSetup, class OutputFunction, class SampleStatusFunction>
void run_sensitivity_analysis(SensiSetup& sensi_setup, OutputFunction output_func, size_t num_runs, size_t num_agents,
                              double tmax, double dt, std::string result_dir, size_t num_runs_per_output,
                              SampleStatusFunction sample_function)
{
#pragma omp barrier
#pragma omp parallel for
    for (size_t run = size_t(0); run < num_runs; ++run) {
        std::cerr << "Start run " << run << "\n" << std::flush;
        std::map<std::string, double> base_values(sensi_setup.base_values);
        std::map<std::string, mio::ParameterDistributionUniform> params(sensi_setup.params);
        draw_base_values(params, base_values);
        ModelSetup setup_base = create_model_setup<ModelSetup>(base_values, tmax, dt, num_agents);
        // calculate base value output
        Model model_base           = create_model<ModelSetup, Model>(setup_base);
        std::vector<double> y_base = output_func(setup_base, model_base, num_runs_per_output, sample_function);
        for (auto it = base_values.begin(); it != base_values.end(); ++it) {
            double old_value = it->second;
            // modify param value by delta
            it->second                  = old_value + sensi_setup.deltas.at(it->first);
            ModelSetup setup_delta      = create_model_setup<ModelSetup>(base_values, tmax, dt, num_agents);
            Model model_delta           = create_model<ModelSetup, Model>(setup_delta);
            std::vector<double> y_delta = output_func(setup_delta, model_delta, num_runs_per_output, sample_function);
            // save elementary effect sample
            for (size_t i = 0; i < y_base.size(); ++i) {
                sensi_setup.elem_effects[i].at(it->first)[run] =
                    (y_delta[i] - y_base[i]) / sensi_setup.deltas.at(it->first);
            }
            // reset param value
            it->second = old_value;
        }
        std::cerr << "End run " << run << "\n" << std::flush;
    }
#pragma omp single
    {
        std::string result_file = result_dir + "_elem_effects";
        save_elementary_effects(sensi_setup.elem_effects, result_file, num_runs);
    }
}

template <class ModelSetup>
double norm_num_infected(mio::TimeSeries<double>& ts, size_t num_regions)
{

    double y = 0;
    //calculate log2 norm of infected
    for (auto t = 0; t < ts.get_num_time_points(); ++t) {
        double num_infected = 0;
        for (size_t region = 0; region < num_regions; ++region) {
            num_infected += ts.get_value(t)[region * static_cast<size_t>(ModelSetup::Status::Count) +
                                            static_cast<size_t>(ModelSetup::Status::I)];
        }
        y += num_infected * num_infected;
    }
    return std::sqrt(y);
}

template <class ModelSetup>
double max_num_infected(mio::TimeSeries<double>& ts, size_t num_regions)
{
    double y = 0;
    //calculate maximum infected over all time points
    for (auto t = 0; t < ts.get_num_time_points(); ++t) {
        double total_num_infected = 0;
        for (size_t region = 0; region < num_regions; ++region) {
            total_num_infected += ts.get_value(t)[region * static_cast<size_t>(ModelSetup::Status::Count) +
                                                  static_cast<size_t>(ModelSetup::Status::I)];
        }
        y = std::max(y, total_num_infected);
    }
    return y;
}

template <class ModelSetup>
double total_transmissions(mio::TimeSeries<double>& ts, size_t num_regions)
{
    double y = 0;
    //calculate total transmissions
    for (auto t = 1; t < ts.get_num_time_points(); ++t) {
        double S_t        = 0;
        double S_t_before = 0;
        for (size_t region = 0; region < num_regions; ++region) {
            S_t += ts.get_value(t)[region * static_cast<size_t>(ModelSetup::Status::Count) +
                                   static_cast<size_t>(ModelSetup::Status::S)];
            S_t_before += ts.get_value(t - 1)[region * static_cast<size_t>(ModelSetup::Status::Count) +
                                              static_cast<size_t>(ModelSetup::Status::S)];
        };
        y += S_t_before - S_t;
    }
    return y;
}

template <class ModelSetup>
double total_deaths(mio::TimeSeries<double>& ts, size_t num_regions)
{
    double y = 0;
    for (size_t region = 0; region < num_regions; ++region) {
        y += ts.get_last_value()[region * static_cast<size_t>(ModelSetup::Status::Count) +
                                 static_cast<size_t>(ModelSetup::Status::D)];
    }
    return y;
}

template <class ModelSetup, class Model, class SampleStatusFunction>
std::vector<double> sensitivity_results(ModelSetup& setup, Model& model, size_t num_runs,
                                        SampleStatusFunction sample_function)
{
    std::vector<mio::TimeSeries<double>> ensemble_results(
        num_runs,
        mio::TimeSeries<double>::zero(setup.tmax, setup.num_regions * static_cast<size_t>(ModelSetup::Status::Count)));
    std::vector<double> timing(num_runs);
    for (size_t run = 0; run < num_runs; ++run) {
        auto sim = mio::Simulation<Model>(model, 0.0, setup.dt);
        sample_function(setup, sim);
        double t_start = omp_get_wtime();
        sim.advance(setup.tmax);
        double t_end     = omp_get_wtime();
        auto& run_result = sim.get_result();

        ensemble_results[run] = mio::interpolate_simulation_result(run_result);
        timing[run]           = t_end - t_start;
    }

    //claculate mean result and mean time
    mio::TimeSeries<double> mean_time_series = std::accumulate(
        ensemble_results.begin(), ensemble_results.end(),
        mio::TimeSeries<double>::zero(ensemble_results[0].get_num_time_points(),
                                      setup.num_regions * static_cast<size_t>(ModelSetup::Status::Count)),
        mio::mpm::paper::add_time_series);
    for (size_t t = 0; t < static_cast<size_t>(mean_time_series.get_num_time_points()); ++t) {
        mean_time_series.get_value(t) *= 1.0 / num_runs;
    }

    double mean_time = (1.0 / num_runs) * std::accumulate(timing.begin(), timing.end(), 0.0);
    mio::unused(mean_time);
    return std::vector<double>{norm_num_infected<ModelSetup>(mean_time_series, setup.num_regions),
                               max_num_infected<ModelSetup>(mean_time_series, setup.num_regions),
                               total_transmissions<ModelSetup>(mean_time_series, setup.num_regions),
                               total_deaths<ModelSetup>(mean_time_series, setup.num_regions)};
}

#endif //SENSITIVITY_ANALYSIS_H
