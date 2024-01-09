#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/weighted_gradient.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "hybrid_paper/library/potentials/potential_germany.h"
#include "mpm/abm.h"
#include "mpm/utility.h"

#include "memilio/io/json_serializer.h"
#include "memilio/data/analyze_result.h"
#include "memilio/utils/time_series.h"

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

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
    using Model = ABM<CommutingPotential<StochastiK, InfectionState>>;
    std::vector<Model::Agent> agents(n_agents);
    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    for (auto& a : agents) {
        Eigen::Vector2d pos_candidate{pos_rng(0.0, double(potential.rows())), pos_rng(0.0, double(potential.cols()))};
        while (metaregions(pos_candidate[0], pos_candidate[1]) == 0 ||
               potential(pos_candidate[0], pos_candidate[1]) != 0) {
            pos_candidate = {pos_rng(0.0, double(potential.rows())), pos_rng(0.0, double(potential.cols()))};
        }
        a.position = {pos_candidate[0], pos_candidate[1]};
        a.region   = metaregions(pos_candidate[0], pos_candidate[1]) - 1;
        a.status   = InfectionState::S;
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(metaregions.maxCoeff(),
                                                                                           metaregions.maxCoeff());
    commute_weights.setZero();

    StochastiK k_provider(commute_weights, metaregions, {metaregions});

    Model model(k_provider, agents, {}, wg.gradient, metaregions, {InfectionState::D},
                std::vector<double>(metaregions.maxCoeff(), 10));
    const double dt   = 0.05;
    double t          = 0;
    const double tmax = 100;

    auto sim = mio::Simulation<Model>(model, t, dt);
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

template <class Model>
void run_multiple_simulation(const std::vector<typename Model::Agent>& agents,
                             std::vector<mio::mpm::AdoptionRate<InfectionState>>& adoption_rates, WeightedGradient& wg,
                             const std::vector<double>& sigma, Eigen::MatrixXi& metaregions, double contact_radius,
                             double tmax, double delta_t, int num_runs, bool save_percentiles)
{

    size_t num_agents  = agents.size();
    size_t num_regions = metaregions.maxCoeff();

    const std::vector<int> county_ids   = {233, 228, 242, 223, 238, 232, 231, 229};
    Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();
    auto ref_pops = std::vector<double>{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(num_regions, num_regions);
    commute_weights.setZero();
    for (int i = 0; i < num_regions; i++) {
        for (int j = 0; j < num_regions; j++) {
            if (i != j) {
                commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
            }
        }
        commute_weights(i, i) = ref_pops[i] - commute_weights.row(i).sum();
    }
    StochastiK k_provider(commute_weights, metaregions, {metaregions});
    Model model(k_provider, agents, adoption_rates, wg.gradient, metaregions, {InfectionState::D}, sigma,
                contact_radius);

    std::vector<mio::TimeSeries<double>> ensemble_results(
        num_runs, mio::TimeSeries<double>::zero(tmax, num_regions * static_cast<size_t>(InfectionState::Count)));
#pragma omp parallel for
    for (int run = 0; run < num_runs; ++run) {
        //std::cerr << "run number: " << run << "\n" << std::flush;
        //std::vector<Model::Agent> agents_run = agents;
        auto run_result       = mio::simulate(0.0, tmax, delta_t, model);
        ensemble_results[run] = mio::interpolate_simulation_result(run_result);
    }

#pragma omp single
    { // add all results
        mio::TimeSeries<double> mean_time_series =
            std::accumulate(ensemble_results.begin(), ensemble_results.end(),
                            mio::TimeSeries<double>::zero(ensemble_results[0].get_num_time_points(),
                                                          num_regions * static_cast<size_t>(InfectionState::Count)),
                            add_time_series);
        //calculate average
        for (size_t t = 0; t < static_cast<size_t>(mean_time_series.get_num_time_points()); ++t) {
            mean_time_series.get_value(t) *= 1.0 / num_runs;
        }

        std::string dir = mio::base_dir() + "cpp/outputs/" + std::to_string(agents.size());

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

void run_single_simulation(std::string init_file, std::vector<mio::mpm::AdoptionRate<InfectionState>>& adoption_rates,
                           WeightedGradient& wg, const std::vector<double>& sigma, Eigen::MatrixXi& metaregions,
                           double tmax, double delta_t)
{
    using Model        = ABM<CommutingPotential<StochastiK, InfectionState>>;
    size_t num_regions = metaregions.maxCoeff();
    std::vector<Model::Agent> agents;
    read_initialization<Model::Agent>(init_file, agents);

    const std::vector<int> county_ids   = {233, 228, 242, 223, 238, 232, 231, 229};
    Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();
    auto ref_pops = std::vector<double>{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(num_regions, num_regions);
    commute_weights.setZero();
    for (int i = 0; i < num_regions; i++) {
        for (int j = 0; j < num_regions; j++) {
            if (i != j) {
                commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
            }
        }
        commute_weights(i, i) = ref_pops[i] - commute_weights.row(i).sum();
    }
    StochastiK k_provider(commute_weights, metaregions, {metaregions});
    Model model(k_provider, agents, {}, wg.gradient, metaregions, {InfectionState::D}, sigma);

    TIME_TYPE sim   = TIME_NOW;
    auto run_result = mio::simulate(0.0, tmax, delta_t, model);
    restart_timer(sim, "# Time for simulation");
    auto interpolated_result = mio::interpolate_simulation_result(run_result);
    FILE* out_file           = fopen("abm_output.txt", "w");
    print_to_file(out_file, run_result, {});
    fclose(out_file);
}

void extrapolate_real_data(Date date, double num_days, double t_Exposed, double t_Carrier, double t_Infected,
                           double mu_C_R)
{
    //number inhabitants per region
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    //county ids
    std::vector<int> regions = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =
        mio::read_confirmed_cases_data("../../data/Germany/cases_all_county_age_ma7.json").value();
    std::sort(confirmed_cases.begin(), confirmed_cases.end(), [](auto&& a, auto&& b) {
        return std::make_tuple(get_region_id(a), a.date) < std::make_tuple(get_region_id(b), b.date);
    });
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
            // std::transform(pop_dists[region].begin(), pop_dists[region].end(), pop_dists[region].begin(),
            //                [&persons_per_agent](auto& c) {
            //                    return std::round(c / persons_per_agent);
            //                });
            for (size_t elem = 0; elem < static_cast<size_t>(InfectionState::Count); ++elem) {
                ts.get_value(t)[region * static_cast<size_t>(InfectionState::Count) + elem] = pop_dists[region][elem];
            }
        }
        date = offset_date_by_days(date, 1);
    }
    FILE* file = fopen((mio::base_dir() + "cpp/outputs/output_extrapolated.txt").c_str(), "w");
    mio::mpm::print_to_file(file, ts, {});
    fclose(file);
}

} //namespace paper
} //namespace mpm
} //namespace mio

int main()
{
    using Status = mio::mpm::paper::InfectionState;
    using Model  = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;

    Eigen::MatrixXi metaregions;
    size_t num_regions = 8;

    WeightedGradient wg(mio::base_dir() + "potentially_germany_grad.json", mio::base_dir() + "boundary_ids.pgm");
    const std::vector<double> weights(14, 500);
    const std::vector<double> sigmas(num_regions, 10);
    wg.apply_weights(weights);

    std::cerr << "Setup: Read metaregions.\n" << std::flush;
    {
        const auto fname = mio::base_dir() + "metagermany.pgm";
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

    mio::Date start_date = mio::Date(2021, 3, 1);

    double t_Exposed                 = 4;
    double t_Carrier                 = 2.67;
    double t_Infected                = 5.03;
    double mu_C_R                    = 0.29;
    double transmission_rate         = 0.25;
    double mu_I_D                    = 0.00476;
    double scaling_factor_trans_rate = 1.0; /// 6.48;
    double contact_radius            = 50;
    double persons_per_agent         = 300;
    double tmax                      = 30;
    double dt                        = 0.1;
    int num_runs                     = 10;

    std::vector<int> regions        = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};

    std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =
        mio::read_confirmed_cases_data(mio::base_dir() + "data/Germany/cases_all_county_age_ma7.json").value();
    std::sort(confirmed_cases.begin(), confirmed_cases.end(), [](auto&& a, auto&& b) {
        return std::make_tuple(get_region_id(a), a.date) < std::make_tuple(get_region_id(b), b.date);
    });

    std::vector<std::vector<double>> pop_dists =
        set_confirmed_case_data(confirmed_cases, regions, populations, start_date, t_Exposed, t_Carrier, t_Infected,
                                mu_C_R)
            .value();

    std::vector<Model::Agent> agents =
        create_agents(pop_dists, populations, persons_per_agent, {metaregions}, false).value();

    std::vector<mio::mpm::AdoptionRate<Status>> adoption_rates;
    for (int i = 0; i < num_regions; ++i) {
        adoption_rates.push_back({Status::S,
                                  Status::E,
                                  mio::mpm::Region(i),
                                  scaling_factor_trans_rate * transmission_rate,
                                  {Status::C, Status::I},
                                  {1, 1}});
        adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(i), 1.0 / t_Exposed});
        adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(i), mu_C_R / t_Carrier});
        adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(i), (1 - mu_C_R) / t_Carrier});
        adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), (1 - mu_I_D) / t_Infected});
        adoption_rates.push_back({Status::I, Status::D, mio::mpm::Region(i), mu_I_D / t_Infected});
    }
    //mio::mpm::paper::extrapolate_real_data(start_date, 30, t_Exposed, t_Carrier, t_Infected, mu_C_R);
    mio::mpm::paper::run_multiple_simulation<Model>(agents, adoption_rates, wg, sigmas, metaregions, contact_radius,
                                                    tmax, dt, num_runs, true);
    //run_single_simulation("init28133.json", adoption_rates, wg, sigmas, metaregions, 100, 0.1);
    //mio::mpm::paper::get_agent_movement(10, metaregions, potential, wg);
    return 0;
}