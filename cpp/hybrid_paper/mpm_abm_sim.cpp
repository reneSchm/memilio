#include "infection_state.h"
#include "weighted_gradient.h"
#include "initialization.h"
#include "mpm/abm.h"
#include "mpm/potentials/potential_germany.h"
#include "memilio/data/analyze_result.h"
#include "hybrid_paper/weighted_gradient.h"
#include "memilio/utils/time_series.h"
#include "mpm/abm.h"
#include "memilio/io/json_serializer.h"
#include "mpm/potentials/potential_germany.h"
#include "hybrid_paper/initialization.h"
#include "memilio/data/analyze_result.h"
#include "mpm/utility.h"
#include <bits/types/FILE.h>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#define TIME_TYPE std::chrono::steady_clock::time_point
#define TIME_NOW std::chrono::steady_clock::now()
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

void get_agent_movement(size_t n_agents, Eigen::MatrixXi& metaregions, Eigen::MatrixXd& potential, WeightedGradient& wg)
{
    using ABM = mio::mpm::ABM<GradientGermany<InfectionState>>;
    std::vector<ABM::Agent> agents(n_agents);
    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    for (auto& a : agents) {
        Eigen::Vector2d pos_candidate{pos_rng(0.0, double(potential.rows())), pos_rng(0.0, double(potential.cols()))};
        while (metaregions(pos_candidate[0], pos_candidate[1]) == 0 ||
               potential(pos_candidate[0], pos_candidate[1]) != 0) {
            pos_candidate = {pos_rng(0.0, double(potential.rows())), pos_rng(0.0, double(potential.cols()))};
        }
        a.position = {pos_candidate[0], pos_candidate[1]};
        a.land     = metaregions(pos_candidate[0], pos_candidate[1]) - 1;
        a.status   = InfectionState::S;
    }

    ABM model(agents, {}, wg.gradient, metaregions);
    const double dt   = 0.05;
    double t          = 0;
    const double tmax = 100;

    auto sim = mio::Simulation<ABM>(model, t, dt);
    while (t < tmax) {
        for (auto& agent : sim.get_model().populations) {
            std::cout << agent.position[0] << " " << agent.position[1] << " ";
            sim.get_model().move(t, dt, agent);
        }
        std::cout << "\n";
        t += dt;
    }
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
    auto num_elements    = static_cast<size_t>(InfectionState::Count);
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
                                            percentile_output.size() * static_cast<size_t>(InfectionState::Count));
    for (Eigen::Index time = 0; time < percentile_output[0].get_num_time_points(); time++) {
        ts.get_time(time) = percentile_output[0].get_time(time);
        for (size_t region = 0; region < percentile_output.size(); ++region) {
            for (Eigen::Index elem = 0; elem < percentile_output[region].get_num_elements(); elem++) {
                ts.get_value(time)[region * static_cast<size_t>(InfectionState::Count) + elem] =
                    percentile_output[region].get_value(time)[elem];
            }
        }
    }

    auto file = fopen(filename.c_str(), "w");
    mio::mpm::print_to_file(file, ts, {});
    fclose(file);
}

void run_multiple_simulation(std::string init_file, std::vector<mio::mpm::AdoptionRate<InfectionState>>& adoption_rates,
                             WeightedGradient& wg, const std::vector<double>& sigma, Eigen::MatrixXi& metaregions,
                             double tmax, double delta_t, int num_runs, bool save_percentiles)
{
    using ABM = mio::mpm::ABM<GradientGermany<InfectionState>>;
    std::vector<ABM::Agent> agents;
    read_initialization<ABM::Agent>(init_file, agents);

    int num_agents = agents.size();

    size_t num_regions = metaregions.maxCoeff();

    std::vector<mio::TimeSeries<double>> ensemble_results(
        num_runs, mio::TimeSeries<double>::zero(tmax, num_regions * static_cast<size_t>(InfectionState::Count)));
    for (int run = 0; run < num_runs; ++run) {
        std::cerr << "run number: " << run << "\n" << std::flush;
        std::vector<ABM::Agent> agents_run = agents;
        ABM model(agents, adoption_rates, wg.gradient, metaregions, {InfectionState::D}, sigma);
        auto run_result       = mio::simulate(0.0, tmax, delta_t, model);
        ensemble_results[run] = mio::interpolate_simulation_result(run_result);
    }

    std::cerr << "runs finished\n";

    // add all results
    mio::TimeSeries<double> mean_time_series = std::accumulate(
        ensemble_results.begin(), ensemble_results.end(),
        mio::TimeSeries<double>::zero(tmax, num_regions * static_cast<size_t>(InfectionState::Count)), add_time_series);
    std::cerr << "ts added\n";
    //calculate average
    for (size_t t = 0; t < static_cast<size_t>(mean_time_series.get_num_time_points()); ++t) {
        mean_time_series.get_value(t) *= 1.0 / num_runs;
    }
    std::cerr << "saving\n";
    //save mean timeseries
    FILE* file = fopen("../../cpp/outputs/output_mean.txt", "w");
    mio::mpm::print_to_file(file, mean_time_series, {});
    fclose(file);
    std::cerr << "saving percentile output\n";
    if (save_percentiles) {
        auto ensemble_percentile = get_format_for_percentile_output(ensemble_results, metaregions.maxCoeff());

        //save percentile output
        auto ensemble_result_p05 = mio::ensemble_percentile(ensemble_percentile, 0.05);
        auto ensemble_result_p25 = mio::ensemble_percentile(ensemble_percentile, 0.25);
        auto ensemble_result_p50 = mio::ensemble_percentile(ensemble_percentile, 0.50);
        auto ensemble_result_p75 = mio::ensemble_percentile(ensemble_percentile, 0.75);
        auto ensemble_result_p95 = mio::ensemble_percentile(ensemble_percentile, 0.95);

        percentile_output_to_file(ensemble_result_p05, "../../cpp/outputs/output_p05.txt");
        percentile_output_to_file(ensemble_result_p25, "../../cpp/outputs/output_p25.txt");
        percentile_output_to_file(ensemble_result_p50, "../../cpp/outputs/output_p50.txt");
        percentile_output_to_file(ensemble_result_p75, "../../cpp/outputs/output_p75.txt");
        percentile_output_to_file(ensemble_result_p95, "../../cpp/outputs/output_p95.txt");
    }
}

void run_single_simulation(std::string init_file, std::vector<mio::mpm::AdoptionRate<InfectionState>>& adoption_rates,
                           WeightedGradient& wg, const std::vector<double>& sigma, Eigen::MatrixXi& metaregions,
                           double tmax, double delta_t)
{
    using ABM = mio::mpm::ABM<GradientGermany<InfectionState>>;
    std::vector<ABM::Agent> agents;
    read_initialization<ABM::Agent>(init_file, agents);
    ABM model(agents, adoption_rates, wg.gradient, metaregions, {InfectionState::D}, sigma);
    TIME_TYPE sim   = TIME_NOW;
    auto run_result = mio::simulate(0.0, tmax, delta_t, model);
    restart_timer(sim, "# Time for simulation");
    auto interpolated_result = mio::interpolate_simulation_result(run_result);
    FILE* out_file           = fopen("abm_output.txt", "w");
    print_to_file(out_file, run_result, {});
    fclose(out_file);
}

void extrapolate_real_data(Date date, double num_days, double t_Exposed, double t_Carrier, double t_Infected,
                           double mu_C_R, double persons_per_agent)
{
    //number inhabitants per region
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    //county ids
    std::vector<int> regions = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =
        mio::read_confirmed_cases_data("../../data/Germany/cases_all_county_age_ma7.json").value();
    mio::TimeSeries<double> ts =
        mio::TimeSeries<double>::zero(num_days, regions.size() * static_cast<size_t>(InfectionState::Count));
    for (size_t t = 0; t < num_days; ++t) {
        //vector with entry for every region. Entries are vector with population for every infection state according to initialization
        std::vector<std::vector<double>> pop_dists =
            set_confirmed_case_data(confirmed_cases, regions, populations, date, t_Exposed, t_Carrier, t_Infected,
                                    mu_C_R)
                .value();
        ts.get_time(t) = static_cast<double>(t);
        for (size_t region = 0; region < regions.size(); ++region) {
            //int num_agents = populations[region] / persons_per_agent;
            std::transform(pop_dists[region].begin(), pop_dists[region].end(), pop_dists[region].begin(),
                           [&persons_per_agent](auto& c) {
                               return std::round(c / persons_per_agent);
                           });
            // while (num_agents > 0) {
            //     auto status = mio::DiscreteDistribution<int>::get_instance()(pop_dists[region]);
            //     ts.get_value(t)[region * static_cast<size_t>(InfectionState::Count) + status]++;
            //     num_agents -= 1;
            // }
            for (size_t elem = 0; elem < static_cast<size_t>(InfectionState::Count); ++elem) {
                ts.get_value(t)[region * static_cast<size_t>(InfectionState::Count) + elem] = pop_dists[region][elem];
            }
        }
        date = offset_date_by_days(date, 1);
    }
    FILE* file = fopen("../../cpp/outputs/output_extrapolated.txt", "w");
    mio::mpm::print_to_file(file, ts, {});
    fclose(file);
}

} //namespace paper
} //namespace mpm
} //namespace mio

int main()
{
    Eigen::MatrixXi metaregions;
    Eigen::MatrixXd potential;

    WeightedGradient wg("../../potentially_germany_grad.json", "../../boundary_ids.pgm");
    const std::vector<double> weights{1000, 800,  850.076, 717.145, 1000, 50,      611.022,
                                      160,  1000, 500,     100,     1200, 725.818, 590.303};
    const std::vector<double> sigmas{15, 29, 15, 15, 28, 35, 43, 30};
    const double slope = 2.0;
    wg.apply_weights(weights);

    const Eigen::Vector2d centre = {wg.gradient.rows() / 2.0, wg.gradient.cols() / 2.0};

    for (Eigen::Index i = 0; i < wg.gradient.rows(); i++) {
        for (Eigen::Index j = 0; j < wg.gradient.cols(); j++) {
            if (wg.base_gradient(i, j) == Eigen::Vector2d{0, 0}) {
                auto direction    = (Eigen::Vector2d{i, j} - centre).normalized();
                wg.gradient(i, j) = slope * direction;
            }
        }
    }

    std::cerr << "Setup: Read potential.\n" << std::flush;
    {
        const auto fname = "../../potentially_germany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            return 1;
        }
        else {
            potential = 8 * mio::mpm::read_pgm(ifile);
            ifile.close();
        }
    }
    std::cerr << "Setup: Read metaregions.\n" << std::flush;
    {
        const auto fname = "../../metagermany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            return 1;
        }
        else {
            metaregions = mio::mpm::read_pgm_raw(ifile).first;
            ifile.close();
        }
    }
    using Status             = mio::mpm::paper::InfectionState;
    double t_Exposed         = 4.2;
    double t_Carrier         = 4.2;
    double t_Infected        = 7.5;
    double mu_C_R            = 0.23;
    double transmission_prob = 0.1;
    double mu_I_D            = 0.01;
    std::vector<mio::mpm::AdoptionRate<Status>> adoption_rates;
    for (int i = 0; i < metaregions.maxCoeff(); ++i) {
        adoption_rates.push_back(
            {Status::S, Status::E, mio::mpm::Region(i), transmission_prob, {Status::C, Status::I}, {1, 1}});
        adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(i), 1.0 / t_Exposed});
        adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(i), mu_C_R / t_Carrier});
        adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(i), (1 - mu_C_R) / t_Carrier});
        adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), (1 - mu_I_D) / t_Infected});
        adoption_rates.push_back({Status::I, Status::D, mio::mpm::Region(i), mu_I_D / t_Infected});
    }
    //run_multiple_simulation("init2809.json", adoption_rates, wg, sigmas, metaregions, 20, 0.1, 5, true);
    mio::mpm::paper::extrapolate_real_data(mio::Date(2021, 3, 1), 100, 4.2, 4.2, 7.5, 0.23, 1000);
    //run_single_simulation("init28133.json", adoption_rates, wg, sigmas, metaregions, 100, 0.1);
    //mio::mpm::paper::get_agent_movement(10, metaregions, potential, wg);
    return 0;
}