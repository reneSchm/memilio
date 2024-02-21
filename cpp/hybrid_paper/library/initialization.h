#ifndef INITIALIZATION_H_
#define INITIALIZATION_H_

#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/metaregion_sampler.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"

#include "memilio/io/epi_data.h"
#include "memilio/io/io.h"
#include "memilio/utils/logging.h"
#include "mpm/abm.h"
#include <cstddef>
#include <map>

namespace mio
{
std::string base_dir();
}

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
                        double t_I, double mu_C_R, double scaling_factor_infected = 1.0,
                        bool set_only_infected = false);

mio::IOResult<std::vector<std::vector<double>>>
set_confirmed_case_data(const std::vector<mio::ConfirmedCasesDataEntry>& confirmed_case_data,
                        const std::vector<int>& regions, const std::vector<double>& populations, mio::Date start_date,
                        double t_E, double t_C, double t_I, double mu_C_R, double scaling_factor_infected = 1.0,
                        bool set_only_infected = false);

std::map<int, double> get_cases_at_date(const std::vector<mio::ConfirmedCasesDataEntry>& confirmed_case_data,
                                        const std::vector<int>& regions, mio::Date date);
std::map<int, double> get_cases_at_date(std::vector<mio::ConfirmedCasesDataEntry>& confirmed_case_data,
                                        const std::vector<int>& regions, mio::Date date);

template <class Agent>
void read_initialization(std::string filename, std::vector<Agent>& agents)
{
    auto result = mio::read_json(filename).value();

    if (!result) {
        mio::log(mio::LogLevel::critical, "Could not open agent initialization file {}", filename);
        exit(1);
    }

    for (size_t i = 0; i < result.size(); ++i) {
        auto a = mio::deserialize_json(result[std::to_string(i)], mio::Tag<Agent>{}).value();
        agents.push_back(Agent{a.position, a.status, a.region});
    }
}

template <class Agent>
mio::IOResult<std::vector<Agent>> create_agents(std::vector<std::vector<double>>& pop_dists,
                                                const std::vector<double>& populations, double persons_per_agent,
                                                const MetaregionSampler& metaregion_sampler, bool save_initialization)
{
    std::vector<Agent> agents;
    for (size_t region = size_t(0); region < pop_dists.size(); ++region) {
        int num_agents = populations[region] / persons_per_agent;
        std::transform(pop_dists[region].begin(), pop_dists[region].end(), pop_dists[region].begin(),
                       [&persons_per_agent](auto& c) {
                           return c / persons_per_agent;
                       });
        while (num_agents > 0) {
            auto status              = mio::DiscreteDistribution<int>::get_instance()(pop_dists[region]);
            Eigen::Vector2d position = metaregion_sampler(region);
            ;
            agents.push_back({position, static_cast<mio::mpm::paper::InfectionState>(status), int(region)});
            pop_dists[region][status] = std::max(0., pop_dists[region][status] - 1);
            num_agents -= 1;
        }
    }

    if (save_initialization) {
        std::string save_path = "init" + std::to_string(agents.size()) + ".json";
        Json::Value all_agents;
        for (size_t i = 0; i < agents.size(); ++i) {
            BOOST_OUTCOME_TRY(agent, mio::serialize_json(agents[i]));
            all_agents[std::to_string(i)] = agent;
        }
        auto write_status = mio::write_json(save_path, all_agents);
    }

    return mio::success(agents);
}

#endif // INITIALIZATION_H_
