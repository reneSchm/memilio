#include "initialization.h"
#include "infection_state.h"
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

    explicit FittingFunctionSetup(const std::vector<int>& regions, const std::vector<double>& inhabitants,
                                  const mio::Date start_date, const double tmax, double dt,
                                  const std::vector<mio::mpm::TransitionRate<Model::Compartments>>& transition_rates,
                                  std::vector<mio::ConfirmedCasesDataEntry>& confirmed_cases)
        : tmax(tmax)
        , dt(dt)
        , regions(regions)
        , inhabitants(inhabitants)
        , start_date(start_date)
        , transition_rates(transition_rates)
        , confirmed_cases(confirmed_cases)
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
    auto result                                             = mio::simulate(0, ffs.tmax, ffs.dt, model);
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
                (1.0 / 3.0) *
                    std::abs(result_t[static_cast<size_t>(Status::I) + static_cast<size_t>(Status::Count) * region] -
                             extrapolated_rki[region][static_cast<size_t>(Status::I)]) +
                (2.0 / 3.0) *
                    std::abs(result_t[static_cast<size_t>(Status::D) + static_cast<size_t>(Status::Count) * region] -
                             extrapolated_rki[region][static_cast<size_t>(Status::D)]);
            l_2[t] += error * error;
        }
        l_2[t] = std::sqrt(l_2[t]);
        date   = mio::offset_date_by_days(date, 1);
    }
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

    // //estimated transition rates
    // std::map<std::tuple<Region, Region>, double> factors{
    //     {{Region(0), Region(1)}, 0.00178656},  {{Region(0), Region(2)}, 0.00241035},
    //     {{Region(0), Region(3)}, 0.0169754},   {{Region(0), Region(4)}, 0.00299189},
    //     {{Region(0), Region(5)}, 0.00046048},  {{Region(0), Region(6)}, 0.000187413},
    //     {{Region(0), Region(7)}, 0.000208213}, {{Region(1), Region(0)}, 0.00175157},
    //     {{Region(1), Region(2)}, 0.000338347}, {{Region(1), Region(3)}, 0.0144762},
    //     {{Region(1), Region(4)}, 0.00243008},  {{Region(1), Region(5)}, 0.00112651},
    //     {{Region(1), Region(6)}, 0.000342293}, {{Region(1), Region(7)}, 0.00025184},
    //     {{Region(2), Region(0)}, 0.00241099},  {{Region(2), Region(1)}, 0.000347093},
    //     {{Region(2), Region(3)}, 0.0108812},   {{Region(2), Region(4)}, 0.00231083},
    //     {{Region(2), Region(5)}, 0.00027616},  {{Region(2), Region(6)}, 0.000186027},
    //     {{Region(2), Region(7)}, 0.000153813}, {{Region(3), Region(0)}, 0.0169618},
    //     {{Region(3), Region(1)}, 0.0144732},   {{Region(3), Region(2)}, 0.0108699},
    //     {{Region(3), Region(4)}, 0.0605866},   {{Region(3), Region(5)}, 0.0122073},
    //     {{Region(3), Region(6)}, 0.0158606},   {{Region(3), Region(7)}, 0.0108883},
    //     {{Region(4), Region(0)}, 0.00300917},  {{Region(4), Region(1)}, 0.00244203},
    //     {{Region(4), Region(2)}, 0.00235371},  {{Region(4), Region(3)}, 0.0604897},
    //     {{Region(4), Region(5)}, 0.00473547},  {{Region(4), Region(6)}, 0.00321984},
    //     {{Region(4), Region(7)}, 0.00504075},  {{Region(5), Region(0)}, 0.000463787},
    //     {{Region(5), Region(1)}, 0.00111669},  {{Region(5), Region(2)}, 0.000285333},
    //     {{Region(5), Region(3)}, 0.0121037},   {{Region(5), Region(4)}, 0.00469013},
    //     {{Region(5), Region(6)}, 0.00480331},  {{Region(5), Region(7)}, 0.00057344},
    //     {{Region(6), Region(0)}, 0.000179413}, {{Region(6), Region(1)}, 0.000350613},
    //     {{Region(6), Region(2)}, 0.000195307}, {{Region(6), Region(3)}, 0.0157696},
    //     {{Region(6), Region(4)}, 0.00321515},  {{Region(6), Region(5)}, 0.00477877},
    //     {{Region(6), Region(7)}, 0.00210827},  {{Region(7), Region(0)}, 0.000217067},
    //     {{Region(7), Region(1)}, 0.0002544},   {{Region(7), Region(2)}, 0.000137067},
    //     {{Region(7), Region(3)}, 0.0107737},   {{Region(7), Region(4)}, 0.00499136},
    //     {{Region(7), Region(5)}, 0.000580373}, {{Region(7), Region(6)}, 0.00212267}};

    //exact transition rates
    std::map<std::tuple<Region, Region>, double> factors{
        {{Region(0), Region(1)}, 0.00177574},  {{Region(0), Region(2)}, 0.00242008},
        {{Region(0), Region(3)}, 0.016983},    {{Region(0), Region(4)}, 0.00301409},
        {{Region(0), Region(5)}, 0.00045548},  {{Region(0), Region(6)}, 0.000183093},
        {{Region(0), Region(7)}, 0.000209522}, {{Region(1), Region(0)}, 0.00177574},
        {{Region(1), Region(2)}, 0.00034445},  {{Region(1), Region(3)}, 0.0144554},
        {{Region(1), Region(4)}, 0.0024382},   {{Region(1), Region(5)}, 0.00110078},
        {{Region(1), Region(6)}, 0.000347833}, {{Region(1), Region(7)}, 0.000251346},
        {{Region(2), Region(0)}, 0.00242008},  {{Region(2), Region(1)}, 0.00034445},
        {{Region(2), Region(3)}, 0.0108871},   {{Region(2), Region(4)}, 0.00232572},
        {{Region(2), Region(5)}, 0.000277803}, {{Region(2), Region(6)}, 0.000194424},
        {{Region(2), Region(7)}, 0.000148419}, {{Region(3), Region(0)}, 0.016983},
        {{Region(3), Region(1)}, 0.0144554},   {{Region(3), Region(2)}, 0.0108871},
        {{Region(3), Region(4)}, 0.0605537},   {{Region(3), Region(5)}, 0.0121034},
        {{Region(3), Region(6)}, 0.0158054},   {{Region(3), Region(7)}, 0.0108538},
        {{Region(4), Region(0)}, 0.00301409},  {{Region(4), Region(1)}, 0.0024382},
        {{Region(4), Region(2)}, 0.00232572},  {{Region(4), Region(3)}, 0.0605537},
        {{Region(4), Region(5)}, 0.00498327},  {{Region(4), Region(6)}, 0.00322363},
        {{Region(4), Region(7)}, 0.00498327},  {{Region(5), Region(0)}, 0.00045548},
        {{Region(5), Region(1)}, 0.00110078},  {{Region(5), Region(2)}, 0.000277803},
        {{Region(5), Region(3)}, 0.0121034},   {{Region(5), Region(4)}, 0.0047219},
        {{Region(5), Region(6)}, 0.00478392},  {{Region(5), Region(7)}, 0.000580243},
        {{Region(6), Region(0)}, 0.000183093}, {{Region(6), Region(1)}, 0.000347833},
        {{Region(6), Region(2)}, 0.000194424}, {{Region(6), Region(3)}, 0.0158054},
        {{Region(6), Region(4)}, 0.00322363},  {{Region(6), Region(5)}, 0.00478392},
        {{Region(6), Region(7)}, 0.00210684},  {{Region(7), Region(0)}, 0.000209522},
        {{Region(7), Region(1)}, 0.000251346}, {{Region(7), Region(2)}, 0.000148419},
        {{Region(7), Region(3)}, 0.0108538},   {{Region(7), Region(4)}, 0.00498327},
        {{Region(7), Region(5)}, 0.000580243}, {{Region(7), Region(6)}, 0.00210684}};

    std::vector<Status> transitioning_states{Status::S, Status::E, Status::C, Status::I, Status::R};
    //No adapted transition behaviour when infected
    std::vector<mio::mpm::TransitionRate<Status>> transition_rates;
    for (auto& rate : factors) {
        for (auto s : transitioning_states) {
            transition_rates.push_back({s, std::get<0>(rate.first), std::get<1>(rate.first), rate.second});
        }
    }
    std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =
        mio::read_confirmed_cases_data("../../data/Germany/cases_all_county_age_ma7.json").value();
    const double tmax    = 30;
    double dt            = 0.1;
    mio::Date start_date = mio::Date(2021, 3, 1);

    const FittingFunctionSetup ffs(regions, populations, start_date, tmax, dt, transition_rates, confirmed_cases);

    std::cout << "Total threads:           " << dlib::default_thread_pool().num_threads_in_pool() << "\n";
    const int num_runs = 10;
    auto result        = dlib::find_min_global(
        dlib::default_thread_pool(),
        [&](double t_Exposed, double t_Carrier, double t_Infected, double mu_C_R, double transmission_rate,
            double mu_I_D) {
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
        {2.67, 2.67, 5, 0.1, 0.25, 0.002}, // lower bounds
        {4, 4, 9, 0.3, 1.25, 0.005}, // upper bounds
        std::chrono::hours(36) // run this long
    );

    std::cout << "Minimizer:\n";
    for (size_t param = 0; param < result.x.size(); ++param) {
        std::cout << result.x(param) << "\n";
    }
    std::cout << "Minimum error:\n" << result.y << "\n";
    return 0;
}
