#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

#include "memilio/io/epi_data.h"
#include "memilio/io/io.h"
#include "hybrid_paper/infection_state.h"
#include "memilio/utils/date.h"
#include "mpm/abm.h"
#include "mpm/potentials/potential_germany.h"
#include "mpm/utility.h"
#include "memilio/io/json_serializer.h"
#include "mpm/potentials/map_reader.h"

//district, county or state id of a data entry if available, 0 (for whole country) otherwise
//used to compare data entries to integer ids in STL algorithms
template <class EpiDataEntry>
int get_region_id(const EpiDataEntry& entry)
{
    return entry.county_id
               ? entry.county_id->get()
               : (entry.state_id ? entry.state_id->get() : (entry.district_id ? entry.district_id->get() : 0));
}
//overload for integers, so the comparison of data entry to integers is symmetric (required by e.g. equal_range)
inline int get_region_id(int id)
{
    return id;
}

// TODO: move this definition to a .cpp file
mio::IOResult<std::vector<std::vector<double>>>
set_confirmed_case_data(std::vector<mio::ConfirmedCasesDataEntry>& confirmed_case_data, const std::vector<int>& regions,
                        const std::vector<double>& populations, mio::Date start_date, double t_E, double t_C,
                        double t_I, double mu_C_R, double scaling_factor_infected = 1.0)
{
    std::vector<double> pop_dist((size_t)mio::mpm::paper::InfectionState::Count);
    std::vector<std::vector<double>> pop_dist_per_region(regions.size(), pop_dist);
    //sort data
    std::sort(confirmed_case_data.begin(), confirmed_case_data.end(), [](auto&& a, auto&& b) {
        return std::make_tuple(get_region_id(a), a.date) < std::make_tuple(get_region_id(b), b.date);
    });

    for (auto region_idx = size_t(0); region_idx < regions.size(); ++region_idx) {

        //get entries for region
        auto region_entry_range_it = std::equal_range(confirmed_case_data.begin(), confirmed_case_data.end(),
                                                      regions[region_idx], [](auto&& a, auto&& b) {
                                                          return get_region_id(a) < get_region_id(b);
                                                      });
        auto region_entry_range    = mio::make_range(region_entry_range_it);

        //TODO: seek correct date(e.g. with find_if) instead of iterating over all
        for (auto&& region_entry : region_entry_range) {
            auto date_df = region_entry.date;
            if (date_df == mio::offset_date_by_days(start_date, 0)) {
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::I] +=
                    scaling_factor_infected * region_entry.num_confirmed;
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::C] -=
                    1 / (1 - mu_C_R) * scaling_factor_infected * region_entry.num_confirmed;
            }
            if (date_df == mio::offset_date_by_days(start_date, t_C)) {
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::C] +=
                    1 / (1 - mu_C_R) * scaling_factor_infected * region_entry.num_confirmed;
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::E] -=
                    1 / (1 - mu_C_R) * scaling_factor_infected * region_entry.num_confirmed;
            }
            if (date_df == mio::offset_date_by_days(start_date, t_E + t_C)) {
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::E] +=
                    1 / (1 - mu_C_R) * scaling_factor_infected * region_entry.num_confirmed;
            }
            if (date_df == mio::offset_date_by_days(start_date, -t_I)) {
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::I] -=
                    scaling_factor_infected * region_entry.num_confirmed;
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::R] +=
                    region_entry.num_recovered;
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::D] += region_entry.num_deaths;
            }
        }
        auto comp_sum =
            std::accumulate(pop_dist_per_region[region_idx].begin(), pop_dist_per_region[region_idx].end(), 0.0);
        pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::S] =
            populations[region_idx] - comp_sum;
    }

    return mio::success(pop_dist_per_region);
}

mio::IOResult<std::vector<std::vector<double>>>
set_confirmed_case_data(const std::vector<mio::ConfirmedCasesDataEntry>& confirmed_case_data,
                        const std::vector<int>& regions, const std::vector<double>& populations, mio::Date start_date,
                        double t_E, double t_C, double t_I, double mu_C_R, double scaling_factor_infected = 1.0)
{
    std::vector<double> pop_dist((size_t)mio::mpm::paper::InfectionState::Count);
    std::vector<std::vector<double>> pop_dist_per_region(regions.size(), pop_dist);

    for (auto region_idx = size_t(0); region_idx < regions.size(); ++region_idx) {

        //get entries for region
        auto region_entry_range_it = std::equal_range(confirmed_case_data.begin(), confirmed_case_data.end(),
                                                      regions[region_idx], [](auto&& a, auto&& b) {
                                                          return get_region_id(a) < get_region_id(b);
                                                      });
        auto region_entry_range    = mio::make_range(region_entry_range_it);

        //TODO: seek correct date(e.g. with find_if) instead of iterating over all
        for (auto&& region_entry : region_entry_range) {
            auto date_df = region_entry.date;
            if (date_df == mio::offset_date_by_days(start_date, 0)) {
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::I] +=
                    scaling_factor_infected * region_entry.num_confirmed;
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::C] -=
                    1 / (1 - mu_C_R) * scaling_factor_infected * region_entry.num_confirmed;
            }
            if (date_df == mio::offset_date_by_days(start_date, t_C)) {
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::C] +=
                    1 / (1 - mu_C_R) * scaling_factor_infected * region_entry.num_confirmed;
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::E] -=
                    1 / (1 - mu_C_R) * scaling_factor_infected * region_entry.num_confirmed;
            }
            if (date_df == mio::offset_date_by_days(start_date, t_E + t_C)) {
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::E] +=
                    1 / (1 - mu_C_R) * scaling_factor_infected * region_entry.num_confirmed;
            }
            if (date_df == mio::offset_date_by_days(start_date, -t_I)) {
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::I] -=
                    scaling_factor_infected * region_entry.num_confirmed;
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::R] +=
                    region_entry.num_recovered;
                pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::D] += region_entry.num_deaths;
            }
        }
        auto comp_sum =
            std::accumulate(pop_dist_per_region[region_idx].begin(), pop_dist_per_region[region_idx].end(), 0.0);
        pop_dist_per_region[region_idx][(size_t)mio::mpm::paper::InfectionState::S] =
            populations[region_idx] - comp_sum;
    }

    return mio::success(pop_dist_per_region);
}

template <class Agent>
void read_initialization(std::string filename, std::vector<Agent>& agents)
{
    auto result = mio::read_json(filename).value();

    if (!result) {
        mio::log(mio::LogLevel::critical, "Could not open agent initialization file {}", filename);
        exit(1);
    }

    for (int i = 0; i < result.size(); ++i) {
        auto a = mio::deserialize_json(result[std::to_string(i)], mio::Tag<Agent>{}).value();
        agents.push_back(Agent{a.position, a.status, a.land});
    }
}

#endif // INITIALIZATION_H_
