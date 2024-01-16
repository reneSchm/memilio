#include "initialization.h"
#include "infection_state.h"

#include "memilio/config.h"
#include "memilio/io/mobility_io.h"

std::string mio::base_dir()
{
    return MEMILIO_BASE_DIR;
}

// load mobility data between all german counties
mio::IOResult<Eigen::MatrixXd> get_transition_matrix_daily_total(std::string data_dir)
{
    BOOST_OUTCOME_TRY(matrix_commuter, mio::read_mobility_plain(data_dir + "/commuter_migration_scaled.txt"));
    BOOST_OUTCOME_TRY(matrix_twitter, mio::read_mobility_plain(data_dir + "/twitter_scaled_1252.txt"));
    Eigen::MatrixXd travel_to_matrix = matrix_commuter + matrix_twitter;
    Eigen::MatrixXd transitions_per_day(travel_to_matrix.rows(), travel_to_matrix.cols());
    for (int from = 0; from < travel_to_matrix.rows(); ++from) {
        for (int to = 0; to < travel_to_matrix.cols(); ++to) {
            transitions_per_day(from, to) = travel_to_matrix(from, to) + travel_to_matrix(to, from);
        }
    }
    return mio::success(transitions_per_day);
}

// load mobility data between all german counties
mio::IOResult<Eigen::MatrixXd> get_transition_matrix(std::string data_dir)
{
    BOOST_OUTCOME_TRY(matrix_commuter, mio::read_mobility_plain(data_dir + "/commuter_migration_scaled.txt"));
    BOOST_OUTCOME_TRY(matrix_twitter, mio::read_mobility_plain(data_dir + "/twitter_scaled_1252.txt"));
    Eigen::MatrixXd transition_matrix = matrix_commuter + matrix_twitter;
    return mio::success(transition_matrix);
}

mio::IOResult<std::vector<std::vector<double>>>
set_confirmed_case_data(std::vector<mio::ConfirmedCasesDataEntry>& confirmed_case_data, const std::vector<int>& regions,
                        const std::vector<double>& populations, mio::Date start_date, double t_E, double t_C,
                        double t_I, double mu_C_R, double scaling_factor_infected)
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
                        double t_E, double t_C, double t_I, double mu_C_R,
                        double scaling_factor_infected) //never fails tehrefore IOResult is not necessary
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

std::map<int, double> get_cases_at_date(const std::vector<mio::ConfirmedCasesDataEntry>& confirmed_case_data,
                                        const std::vector<int>& regions, mio::Date date)
{
    std::map<int, double> cases;
    for (auto& region_id : regions) {
        //get entries for region
        auto region_entry_range_it =
            std::equal_range(confirmed_case_data.begin(), confirmed_case_data.end(), region_id, [](auto&& a, auto&& b) {
                return get_region_id(a) < get_region_id(b);
            });
        auto region_entry_range = mio::make_range(region_entry_range_it);
        for (auto&& entry : region_entry_range) {
            if (date == entry.date) {
                cases[region_id] = entry.num_confirmed;
                break;
            }
        }
    }
    return cases;
}