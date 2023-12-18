#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

#include "memilio/io/epi_data.h"
#include "memilio/io/io.h"
#include "memilio/utils/logging.h"

// load mobility data between all german counties
mio::IOResult<Eigen::MatrixXd> get_transition_matrix_daily_total(std::string data_dir);

// load mobility data between all german counties
mio::IOResult<Eigen::MatrixXd> get_transition_matrix(std::string data_dir);

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

mio::IOResult<std::vector<std::vector<double>>>
set_confirmed_case_data(std::vector<mio::ConfirmedCasesDataEntry>& confirmed_case_data, const std::vector<int>& regions,
                        const std::vector<double>& populations, mio::Date start_date, double t_E, double t_C,
                        double t_I, double mu_C_R, double scaling_factor_infected = 1.0);

mio::IOResult<std::vector<std::vector<double>>>
set_confirmed_case_data(const std::vector<mio::ConfirmedCasesDataEntry>& confirmed_case_data,
                        const std::vector<int>& regions, const std::vector<double>& populations, mio::Date start_date,
                        double t_E, double t_C, double t_I, double mu_C_R, double scaling_factor_infected = 1.0);

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
