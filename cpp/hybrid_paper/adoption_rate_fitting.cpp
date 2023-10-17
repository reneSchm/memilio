#include "models/mpm/pdmm.h"
#include "hybrid_paper/weighted_potential.h"
#include "hybrid_paper/initialization.h"
#include "mpm/region.h"
#include "memilio/data/analyze_result.h"

#include <dlib/global_optimization.h>
#include <omp.h>
#include <map>

#include <set>

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()
#define PRECISION 17

struct FittingFunctionSetup {
    using Model = mio::mpm::PDMModel<8, mio::mpm::paper::InfectionState>;

    //mio::TimeSeries<double> extrapolated_real_data;
    double tmax;
    std::vector<std::vector<ScalarType>> populations;
    std::vector<int> regions;
    std::vector<double> inhabitants;
    mio::Date start_date;
    double tmax;
    std::vector<mio::mpm::TransitionRate<Model::Compartments>> transition_rates;

    explicit FittingFunctionSetup(const std::vector<int>& regions, const std::vector<double>& inhabitants,
                                  const mio::Date start_date, const double tmax,
                                  const std::vector<mio::mpm::TransitionRate<Model::Compartments>>& transition_rates)
        : regions(regions)
        , inhabitants(inhabitants)
        , start_date(start_date)
        , tmax(tmax)
        , transition_rates(transition_rates)
    {
    }
};

double single_run_infections_deaths_error(const FittingFunctionSetup& ffs, double t_Exposed, double t_Carrier,
                                          double t_Infected, double mu_C_R, double transmission_prob, double mu_I_D)
{
    using Model  = FittingFunctionSetup::Model;
    using Status = FittingFunctionSetup::Model::Compartments;

    Model model;

    //vector with entry for every region. Entries are vector with population for every infection state according to initialization
    std::vector<std::vector<double>> pop_dists =
        set_confirmed_case_data(ffs.regions, ffs.inhabitants, ffs.start_date, t_Exposed, t_Carrier, t_Infected, mu_C_R)
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
    for(int i=0; i<ffs.regions.size(); ++i){
        adoption_rates.push_back({Status::S, Status::E, mio::mpm::Region(i), transmission_prob, {Status::C, Status::I}, {1, 1}});
        adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(i), 1.0/t_Exposed});
        adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(i), mu_C_R / t_Carrier});
        adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(i), (1-mu_C_R) / t_Carrier});
        adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), (1 - mu_I_D) / t_Infected});
        adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), mu_I_D / t_Infected});
    }
    model.parameters.get<mio::mpm::AdoptionRates<Status>>() = adoption_rates;
    auto result = mio::simulate(0, ffs.tmax, 0.1, model);
    //interpolate result to full days
    auto interpolated_result = mio::interpolate_simulation_result(result);

    mio::Date date = ffs.start_date;
    //calculate and return error
    std::vector<double> l_2(interpolated_result.get_num_time_points());
    for(size_t t=0; t<interpolated_result.get_num_time_points(); ++t){
        auto extrapolated_rki = set_confirmed_case_data(ffs.regions, ffs.inhabitants, date, t_Exposed, t_Carrier, t_Infected, mu_C_R)
            .value();
        auto result_t = interpolated_result.get_value(t);
        assert(result_t.size() == (extrapolated_rki.size() * static_cast<size_t>(Status::Count)));
        //calc error for every region
        for(size_t region=0; region<extrapolated_rki.size(); ++region){
            for(size_t comp=0; comp<static_cast<size_t>(Status::Count); ++comp){
                auto error = std::abs(result_t[comp + static_cast<size_t>(Status::Count)*region] - extrapolated_rki[region][comp]);
                l_2[t] += error * error;
            }
        }
        //average over all regions
        l_2[t] = std::sqrt(l_2[t]);
        l_2[t] /= extrapolated_rki.size();        
        date = mio::offset_date_by_days(date, 1);
    }
    //return mean over all timesteps
    return std::accumulate(l_2.begin(), l_2.end(), 0.0) / l_2.size();;
}

int main()
{
    using state  = mio::mpm::paper::InfectionState;
    using Region = mio::mpm::Region;
    //number inhabitants per region
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    //county ids
    std::vector<int> regions = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    //3 kommastelen verschieben
    double kappa = 0.001;
    //transition rates according to parameter estimation
    std::map<std::tuple<Region, Region>, double> factors{{{Region(0), Region(4)}, 0.00324262 * kappa},
                                                         {{Region(1), Region(0)}, 0.0000453513 * kappa},
                                                         {{Region(2), Region(0)}, 0.00119682 * kappa},
                                                         {{Region(2), Region(4)}, 0.00661358 * kappa},
                                                         {{Region(3), Region(0)}, 0.0003551 * kappa},
                                                         {{Region(3), Region(1)}, 0.0277554 * kappa},
                                                         {{Region(3), Region(4)}, 0.0202192},
                                                         {{Region(4), Region(0)}, 0.0772854 * kappa},
                                                         {{Region(4), Region(1)}, 0.171165 * kappa},
                                                         {{Region(4), Region(2)}, 0.00562129 * kappa},
                                                         {{Region(4), Region(3)}, 0.0201224},
                                                         {{Region(4), Region(5)}, 0.000231291 * kappa},
                                                         {{Region(4), Region(6)}, 0.00019365 * kappa},
                                                         {{Region(4), Region(7)}, 0.00101667},
                                                         {{Region(5), Region(1)}, 0.0000453513 * kappa},
                                                         {{Region(5), Region(4)}, 0.0000453513 * kappa},
                                                         {{Region(6), Region(4)}, 0.000192289 * kappa},
                                                         {{Region(6), Region(5)}, 0.000784123 * kappa},
                                                         {{Region(6), Region(7)}, 0.00134482},
                                                         {{Region(7), Region(4)}, 0.00101442},
                                                         {{Region(7), Region(6)}, 0.00138364}};
    std::vector<state> transitioning_states{state::S, state::E, state::C, state::I, state::R};
    //No adapted transition behaviour when infected
    std::vector<mio::mpm::TransitionRate<state>> transition_rates;
    for (auto& rate : factors) {
        for (auto s : transitioning_states) {
            transition_rates.push_back({s, std::get<0>(rate.first), std::get<1>(rate.first), rate.second});
        }
    }
    FittingFunctionSetup ffs(regions, populations, mio::Date(2020, 12, 12), 100, transition_rates);
    return 0;
}