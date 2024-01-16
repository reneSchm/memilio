#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/infection_state.h"
#include "mpm/pdmm.h"
#include "mpm/region.h"
#include "memilio/data/analyze_result.h"
#include "memilio/io/epi_data.h"

#include <dlib/global_optimization.h>
#include <omp.h>
#include <map>

#include <set>

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()
#define PRECISION 17

#define restart_timer(timer, description)                                                                              \
    {                                                                                                                  \
        TIME_TYPE new_time = TIME_NOW;                                                                                 \
        std::cout << "\r" << description << " :: " << PRINTABLE_TIME(new_time - timer) << std::endl << std::flush;     \
        timer = TIME_NOW;                                                                                              \
    }

struct FittingFunctionSetup {
    using Model = mio::mpm::PDMModel<8, mio::mpm::paper::InfectionState>;

    //mio::TimeSeries<double> extrapolated_real_data;
    double tmax;
    double dt;
    std::vector<int> regions;
    std::vector<double> inhabitants;
    mio::Date start_date;
    std::vector<mio::mpm::TransitionRate<Model::Compartments>> transition_rates;
    std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases;
    std::vector<mio::ConfirmedCasesDataEntry> confirmed_new_infections;

    explicit FittingFunctionSetup(const std::vector<int>& regions, const std::vector<double>& inhabitants,
                                  const mio::Date start_date, const double tmax, double dt,
                                  const std::vector<mio::mpm::TransitionRate<Model::Compartments>>& transition_rates,
                                  std::vector<mio::ConfirmedCasesDataEntry>& confirmed_cases,
                                  std::vector<mio::ConfirmedCasesDataEntry>& confirmed_new_infections)
        : tmax(tmax)
        , dt(dt)
        , regions(regions)
        , inhabitants(inhabitants)
        , start_date(start_date)
        , transition_rates(transition_rates)
        , confirmed_cases(confirmed_cases)
        , confirmed_new_infections(confirmed_new_infections)
    {
        std::sort(confirmed_cases.begin(), confirmed_cases.end(), [](auto&& a, auto&& b) {
            return std::make_tuple(get_region_id(a), a.date) < std::make_tuple(get_region_id(b), b.date);
        });
    }
};

double single_run_infection_state_error(const FittingFunctionSetup& ffs, double t_Exposed, double t_Carrier,
                                        double t_Infected, double mu_C_R, double transmission_rate, double mu_I_D)
{
    using Model  = FittingFunctionSetup::Model;
    using Status = FittingFunctionSetup::Model::Compartments;

    Model model;

    //vector with entry for every region. Entries are vector with population for every infection state according to initialization
    std::vector<std::vector<double>> pop_dists =
        set_confirmed_case_data(ffs.confirmed_cases, ffs.regions, ffs.inhabitants, ffs.start_date, t_Exposed, t_Carrier,
                                t_Infected, mu_C_R)
            .value();

    //set populations for model
    for (size_t k = 0; k < ffs.regions.size(); ++k) {
        for (int i = 0; i < static_cast<size_t>(Status::Count); ++i) {
            model.populations[{static_cast<mio::mpm::Region>(k), static_cast<Status>(i)}] = pop_dists[k][i];
        }
    }
    //set transion rates for model (according to parameter estimation)
    model.parameters.get<mio::mpm::TransitionRates<Status>>() = ffs.transition_rates;
    //set adoption rates according to given parameters
    std::vector<mio::mpm::AdoptionRate<Status>> adoption_rates;
    for (int i = 0; i < ffs.regions.size(); ++i) {
        adoption_rates.push_back(
            {Status::S, Status::E, mio::mpm::Region(i), transmission_rate, {Status::C, Status::I}, {1, 1}});
        adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(i), 1.0 / t_Exposed});
        adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(i), mu_C_R / t_Carrier});
        adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(i), (1 - mu_C_R) / t_Carrier});
        adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), (1 - mu_I_D) / t_Infected});
        adoption_rates.push_back({Status::I, Status::D, mio::mpm::Region(i), mu_I_D / t_Infected});
    }
    model.parameters.get<mio::mpm::AdoptionRates<Status>>() = adoption_rates;
#ifdef USE_NEW_INFECTIONS
    auto sim = mio::Simulation<Model>(model, 0.0, ffs.dt);
    sim.advance(ffs.tmax);
    //accumulate flows
    auto flow_ts            = *(sim.get_model().all_flows);
    auto interpolated_flows = mio::interpolate_simulation_result(flow_ts);
    mio::TimeSeries<double> flows(flow_ts.get_num_elements());
    size_t t     = 0;
    size_t t_int = 0;
    for (size_t t_int = 0; t_int < interpolated_flows.get_num_time_points(); ++t_int) {
        Eigen::VectorXd acc_flows = Eigen::VectorXd::Zero(interpolated_flows.get_num_elements());
        while (flow_ts.get_time(t) < interpolated_flows.get_time(t_int)) {
            acc_flows += flow_ts.get_value(t);
            ++t;
        }
        acc_flows += interpolated_flows.get_value(t_int);
        if (t_int > 0) {
            acc_flows -= interpolated_flows.get_value(t_int - 1);
        }
        flows.add_time_point(interpolated_flows.get_time(t_int), acc_flows);
        if (flow_ts.get_time(t) == interpolated_flows.get_time(t_int)) {
            ++t;
        }
    }
    mio::Date date = mio::offset_date_by_days(ffs.start_date, 1);
    std::vector<double> l_2(static_cast<size_t>(ffs.tmax));
    for (size_t d = 0; d < ffs.tmax; ++d) {
        auto confirmed_per_region = get_cases_at_date(ffs.confirmed_new_infections, ffs.regions, date);
        for (size_t region = 0; region < ffs.regions.size(); ++region) {
            auto flows_simulated  = flows.get_value(d);
            double new_infections = 0.1 * flows_simulated[get_region_flow_index(region, Status::E, Status::C)] +
                                    flows_simulated[get_region_flow_index(region, Status::C, Status::I)];
            auto error = std::abs(confirmed_per_region.at(ffs.regions[region]) - new_infections);
            l_2[d] += error * error;
        }
        l_2[t] = std::sqrt(l_2[t]);
        date   = mio::offset_date_by_days(date, 1);
    }
#else
    auto result = mio::simulate(0, ffs.tmax, ffs.dt, model);
    //interpolate result to full days
    auto interpolated_result = mio::interpolate_simulation_result(result);

    mio::Date date = ffs.start_date;
    //calculate and return error
    std::vector<double> l_2(interpolated_result.get_num_time_points());
    for (size_t t = 0; t < interpolated_result.get_num_time_points(); ++t) {
        auto extrapolated_rki = set_confirmed_case_data(ffs.confirmed_cases, ffs.regions, ffs.inhabitants, date,
                                                        t_Exposed, t_Carrier, t_Infected, mu_C_R)
                                    .value();
        auto result_t = interpolated_result.get_value(t);
        assert(result_t.size() == (extrapolated_rki.size() * static_cast<size_t>(Status::Count)));
        //calc error for every region
        for (size_t region = 0; region < extrapolated_rki.size(); ++region) {
            auto error =
                (2.0 / 3.0) *
                    std::abs(result_t[static_cast<size_t>(Status::I) + static_cast<size_t>(Status::Count) * region] -
                             extrapolated_rki[region][static_cast<size_t>(Status::I)]) +
                (1.0 / 3.0) *
                    std::abs(result_t[static_cast<size_t>(Status::D) + static_cast<size_t>(Status::Count) * region] -
                             extrapolated_rki[region][static_cast<size_t>(Status::D)]);
            l_2[t] += error * error;
        }
        l_2[t] = std::sqrt(l_2[t]);
        date   = mio::offset_date_by_days(date, 1);
    }
#endif
    //return mean over all timesteps
    return std::accumulate(l_2.begin(), l_2.end(), 0.0) / l_2.size();
}

double average_run_infection_state_error(const FittingFunctionSetup& ffs, double t_Exposed, double t_Carrier,
                                         double t_Infected, double mu_C_R, double transmission_rate, double mu_I_D,
                                         int num_runs)
{
    TIME_TYPE run = TIME_NOW;
    std::vector<double> errors(num_runs);
    for (int run = 0; run < num_runs; ++run) {
        errors[run] =
            single_run_infection_state_error(ffs, t_Exposed, t_Carrier, t_Infected, mu_C_R, transmission_rate, mu_I_D);
    }
    restart_timer(run, "fitting_time");
    return std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
}

int main()
{
    mio::set_log_level(mio::LogLevel::warn);
    dlib::mutex print_mutex;
    using Status = mio::mpm::paper::InfectionState;
    using Region = mio::mpm::Region;
    //number inhabitants per region
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    //county ids
    std::vector<int> regions = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};

    //estimated transition rates
    std::map<std::tuple<Region, Region>, double> factors{
        {{Region(0), Region(1)}, 0.000958613}, {{Region(0), Region(2)}, 0.00172736},
        {{Region(0), Region(3)}, 0.0136988},   {{Region(0), Region(4)}, 0.00261568},
        {{Region(0), Region(5)}, 0.000317227}, {{Region(0), Region(6)}, 0.000100373},
        {{Region(0), Region(7)}, 0.00012256},  {{Region(1), Region(0)}, 0.000825387},
        {{Region(1), Region(2)}, 0.00023648},  {{Region(1), Region(3)}, 0.0112213},
        {{Region(1), Region(4)}, 0.00202101},  {{Region(1), Region(5)}, 0.00062912},
        {{Region(1), Region(6)}, 0.000201067}, {{Region(1), Region(7)}, 0.000146773},
        {{Region(2), Region(0)}, 0.000712533}, {{Region(2), Region(1)}, 0.000102613},
        {{Region(2), Region(3)}, 0.00675979},  {{Region(2), Region(4)}, 0.00160171},
        {{Region(2), Region(5)}, 0.000175467}, {{Region(2), Region(6)}, 0.00010336},
        {{Region(2), Region(7)}, 6.21867e-05}, {{Region(3), Region(0)}, 0.00329632},
        {{Region(3), Region(1)}, 0.00322347},  {{Region(3), Region(2)}, 0.00412565},
        {{Region(3), Region(4)}, 0.0332566},   {{Region(3), Region(5)}, 0.00462197},
        {{Region(3), Region(6)}, 0.00659424},  {{Region(3), Region(7)}, 0.00255147},
        {{Region(4), Region(0)}, 0.000388373}, {{Region(4), Region(1)}, 0.000406827},
        {{Region(4), Region(2)}, 0.000721387}, {{Region(4), Region(3)}, 0.027394},
        {{Region(4), Region(5)}, 0.00127328},  {{Region(4), Region(6)}, 0.00068224},
        {{Region(4), Region(7)}, 0.00104491},  {{Region(5), Region(0)}, 0.00013728},
        {{Region(5), Region(1)}, 0.000475627}, {{Region(5), Region(2)}, 0.00010688},
        {{Region(5), Region(3)}, 0.00754293},  {{Region(5), Region(4)}, 0.0034704},
        {{Region(5), Region(6)}, 0.00210027},  {{Region(5), Region(7)}, 0.000226667},
        {{Region(6), Region(0)}, 7.264e-05},   {{Region(6), Region(1)}, 0.0001424},
        {{Region(6), Region(2)}, 9.55733e-05}, {{Region(6), Region(3)}, 0.00921109},
        {{Region(6), Region(4)}, 0.0025216},   {{Region(6), Region(5)}, 0.00266944},
        {{Region(6), Region(7)}, 0.00156053},  {{Region(7), Region(0)}, 7.81867e-05},
        {{Region(7), Region(1)}, 0.0001024},   {{Region(7), Region(2)}, 8.256e-05},
        {{Region(7), Region(3)}, 0.00833152},  {{Region(7), Region(4)}, 0.00393717},
        {{Region(7), Region(5)}, 0.000354987}, {{Region(7), Region(6)}, 0.00055456}};

    // size_t num_regions                  = 8;
    // const std::vector<int> county_ids   = {233, 228, 242, 223, 238, 232, 231, 229};
    // Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();
    // auto total_population               = std::accumulate(populations.begin(), populations.end(), 0.0);
    // Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(num_regions, num_regions);
    // commute_weights.setZero();
    // for (int i = 0; i < num_regions; i++) {
    //     for (int j = 0; j < num_regions; j++) {
    //         if (i != j) {
    //             commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
    //         }
    //     }
    //     commute_weights(i, i) = populations[i] - commute_weights.row(i).sum();
    // }
    // //exact transition rates
    // std::map<std::tuple<Region, Region>, double> factors;
    // for (int i = 0; i < num_regions; i++) {
    //     for (int j = 0; j < num_regions; j++) {
    //         if (i != j) {
    //             factors.insert({{Region(i), Region(j)}, commute_weights(i, j) / total_population});
    //         }
    //     }
    // }
    std::vector<Status> transitioning_states{
        Status::S, Status::E, Status::C,
        Status::I /*, Status::R*/}; //moving agents of status R is not relevant for fitting
    //No adapted transition behaviour when infected
    std::vector<mio::mpm::TransitionRate<Status>> transition_rates;
    for (auto& rate : factors) {
        for (auto s : transitioning_states) {
            transition_rates.push_back({s, std::get<0>(rate.first), std::get<1>(rate.first), rate.second});
        }
    }
    std::vector<mio::ConfirmedCasesDataEntry> confirmed_new_infections =
        mio::read_confirmed_cases_data(mio::base_dir() + "/data/Germany/new_infections_all_county_ma7.json").value();
    std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =
        mio::read_confirmed_cases_data(mio::base_dir() + "/data/Germany/cases_all_county_age_ma7.json").value();
    const double tmax    = 30;
    double dt            = 0.1;
    mio::Date start_date = mio::Date(2021, 3, 1);
    //mio::Date start_date = mio::Date(2021, 4, 1);
    const FittingFunctionSetup ffs(regions, populations, start_date, tmax, dt, transition_rates, confirmed_cases,
                                   confirmed_new_infections);

    std::cout << "Total threads:           " << dlib::default_thread_pool().num_threads_in_pool() << "\n";
    const int num_runs = 10;
    auto result = dlib::find_min_global(dlib::default_thread_pool(),
        [&](double t_Exposed, double t_Carrier, double t_Infected, double mu_C_R, double transmission_rate/*,
            double mu_I_D*/) {
            double mu_I_D = 0.0;
            // calculate error
            auto err = average_run_infection_state_error(ffs, t_Exposed, t_Carrier, t_Infected, mu_C_R,
                                                         transmission_rate, mu_I_D, num_runs);

            dlib::auto_mutex lock_printing_until_end_of_scope(print_mutex);
            std::cerr << " t_E: " << t_Exposed << ", t_C: " << t_Carrier << ", t_I: " << t_Infected << "\n";
            std::cerr << " mu_C_R: " << mu_C_R << ", trans_prob: " << transmission_rate << ", mu_I_D: " << mu_I_D
                      << "\n";
            std::cerr << "E: " << err << "\n\n";
            return err;
        },
        {2.67, 2.67, 5, 0.1, 0.25/*, 0.002*/}, // lower bounds
        {4, 4, 9, 0.3, 1.25/*, 0.005*/}, // upper bounds
        std::chrono::seconds(3) // run this long
    );

    std::cout << "Minimizer:\n";
    for (size_t param = 0; param < result.x.size(); ++param) {
        std::cout << result.x(param) << "\n";
    }
    std::cout << "Minimum error:\n" << result.y << "\n";
    return 0;
}
