#include "hybrid_paper/library/analyze_result.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/munich/munich_setup.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "mpm/abm.h"
#include "mpm/pdmm.h"
#include "mpm/region.h"
#include "memilio/data/analyze_result.h"
#include "memilio/io/epi_data.h"

#include <dlib/global_optimization.h>
#include <omp.h>
#include <map>

#include <ostream>
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

using MunichSetup =
    mio::mpm::paper::MunichSetup<mio::mpm::ABM<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>>::Agent>;

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

double single_run_infection_state_error(const MunichSetup& setup, const std::vector<double>& transmission_rate,
                                        const std::vector<mio::ConfirmedCasesDataEntry>& confirmed_new_infections)
{
    const int num_regions = 8;
    using Model           = mio::mpm::PDMModel<num_regions, MunichSetup::Status>;
    using Status          = Model::Compartments;

    Model model = setup.create_pdmm<Model>();

    //set adoption rates according to given parameters
    std::vector<mio::mpm::AdoptionRate<Status>> adoption_rates;
    for (int i = 0; i < num_regions; ++i) {
        adoption_rates.push_back(
            {Status::S, Status::E, mio::mpm::Region(i), transmission_rate[i], {Status::C, Status::I}, {1, 1}});
        adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(i), 1.0 / setup.t_Exposed});
        adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(i), setup.mu_C_R / setup.t_Carrier});
        adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(i), (1 - setup.mu_C_R) / setup.t_Carrier});
        adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), (1 - setup.mu_I_D) / setup.t_Infected});
        adoption_rates.push_back({Status::I, Status::D, mio::mpm::Region(i), setup.mu_I_D / setup.t_Infected});
    }
    model.parameters.get<mio::mpm::AdoptionRates<Status>>() = adoption_rates;

#ifdef USE_NEW_INFECTIONS
    auto sim = mio::Simulation<Model>(model, 0.0, setup.dt);
    sim.advance(setup.tmax);

    auto accumulated_flows = mio::mpm::accumulate_flows(*sim.get_model().all_flows);

    mio::Date date = mio::offset_date_by_days(setup.start_date, 1);
    std::vector<double> l_2(static_cast<size_t>(std::ceil(setup.tmax)));
    for (size_t d = 0; d < setup.tmax; ++d) {
        auto confirmed_per_region = get_cases_at_date(confirmed_new_infections, setup.region_ids, date);
        auto flows_simulated      = accumulated_flows.get_value(d);
        for (size_t region = 0; region < num_regions; ++region) {
            double new_infections = 0.1 * flows_simulated[get_region_flow_index(region, Status::E, Status::C)] +
                                    flows_simulated[get_region_flow_index(region, Status::C, Status::I)];
            auto error =
                std::abs(confirmed_per_region.at(setup.region_ids[region]) - new_infections * setup.persons_per_agent);
            l_2[d] += error * error;
        }
        l_2[d] = std::sqrt(l_2[d]);
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

double average_run_infection_state_error(const MunichSetup& setup, const std::vector<double>& transmission_rate,
                                         const std::vector<mio::ConfirmedCasesDataEntry>& confirmed_new_infections,
                                         int num_runs)
{
    // TIME_TYPE run = TIME_NOW;
    std::vector<double> errors(num_runs);
    for (int run = 0; run < num_runs; ++run) {
        errors[run] = single_run_infection_state_error(setup, transmission_rate, confirmed_new_infections);
    }
    // restart_timer(run, "fitting_time");
    return std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
}

int main()
{
    mio::set_log_level(mio::LogLevel::warn);
    dlib::mutex print_mutex;
    using Status = mio::mpm::paper::InfectionState;
    using Region = mio::mpm::Region;
    using ABM    = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;

    const double solver_epsilon   = 1;
    const auto total_fitting_time = std::chrono::hours(16 + 5 * 24);
    const int num_runs            = 10;
    auto total_num_threads        = dlib::default_thread_pool().num_threads_in_pool();

    std::cout << "Current path is          " << fs::current_path()
              << "\n"
              //   << "ABM tmax:                " << tmax << "\n"
              << "Total threads:           " << total_num_threads << "\n"
              << "Total fitting time:      " << PRINTABLE_TIME(total_fitting_time) << "\n"
              << "Solver epsilon:          " << solver_epsilon << "\n"
              << "\n"
              << std::flush;

    std::vector<mio::ConfirmedCasesDataEntry> confirmed_new_infections =
        mio::read_confirmed_cases_data(mio::base_dir() + "/data/Germany/new_infections_all_county_ma7.json").value();

    std::sort(confirmed_new_infections.begin(), confirmed_new_infections.end(), [](auto&& a, auto&& b) {
        return std::make_tuple(get_region_id(a), a.date) < std::make_tuple(get_region_id(b), b.date);
    });

    MunichSetup setup{};
    setup.transition_rates.clear();

    // const FittingFunctionSetup ffs(regions, populations, start_date, tmax, dt, transition_rates, confirmed_cases,
    //    confirmed_new_infections);

    std::cout << "Setup Finished. Starting Fitting.\n" << std::flush;

    double min     = 10e10;
    double min_num = 0;

    auto result = dlib::find_min_global(
        dlib::default_thread_pool(),
        [&](double tr_0, double tr_1, double tr_2, double tr_3, double tr_4, double tr_5, double tr_6, double tr_7) {
            // calculate error
            auto err = average_run_infection_state_error(setup, {tr_0, tr_1, tr_2, tr_3, tr_4, tr_5, tr_6, tr_7},
                                                         confirmed_new_infections, num_runs);

            dlib::auto_mutex lock_printing_until_end_of_scope(print_mutex);
            std::cerr << "transmission_rates: " << tr_0 << ", " << tr_1 << ", " << tr_2 << ", " << tr_3 << ", " << tr_4
                      << ", " << tr_5 << ", " << tr_6 << ", " << tr_7 << "\n";
            std::cerr << "E: " << err << "\n";
            if (err < min) {
                min = err;
                std::cout << "Current Minimizer[" << min_num << "] @ E: " << err << "\n";
                std::cout << "transmission_rates: " << tr_0 << ", " << tr_1 << ", " << tr_2 << ", " << tr_3 << ", "
                          << tr_4 << ", " << tr_5 << ", " << tr_6 << ", " << tr_7 << "\n";
                min_num++;
            }
            std::cerr << "\n";
            return err;
        },
        {0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01}, // lower bounds
        {0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50}, // upper bounds
        total_fitting_time, // run this long
        solver_epsilon);

    std::cout << "Minimizer:\n";
    for (size_t param = 0; param < result.x.size(); ++param) {
        std::cout << result.x(param) << "\n";
    }
    std::cout << "Minimum error:\n" << result.y << "\n";
    return 0;
}
