#include "memilio/io/epi_data.h"
#include "memilio/io/io.h"
#include "hybrid_paper/infection_state.h"
#include "memilio/utils/date.h"
#include "mpm/abm.h"
#include "mpm/potentials/potential_germany.h"
#include "mpm/utility.h"
#include "memilio/io/json_serializer.h"
#include "mpm/potentials/map_reader.h"

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()

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
int get_region_id(int id)
{
    return id;
}

mio::IOResult<std::vector<std::vector<double>>>
set_confirmed_case_data(std::vector<int>& regions, std::vector<double>& populations, mio::Date start_date, double t_E,
                        double t_C, double t_I, double mu_C_R, double scaling_factor_infected = 1.0)
{
    std::vector<double> pop_dist((size_t)mio::mpm::paper::InfectionState::Count);
    std::vector<std::vector<double>> pop_dist_per_region(regions.size(), pop_dist);
    BOOST_OUTCOME_TRY(confirmed_case_data,
                      mio::read_confirmed_cases_data("../../data/Germany/cases_all_county_age_ma7.json"));
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

mio::IOResult<std::vector<mio::mpm::ABM<PotentialGermany<mio::mpm::paper::InfectionState>>::Agent>>
create_agents(std::vector<std::vector<double>>& pop_dists, std::vector<double>& populations, double persons_per_agent,
              Eigen::MatrixXi& metaregions, bool save_initialization)
{
    std::vector<mio::mpm::ABM<PotentialGermany<mio::mpm::paper::InfectionState>>::Agent> agents;
    for (size_t region = size_t(0); region < pop_dists.size(); ++region) {
        int num_agents = populations[region] / persons_per_agent;
        std::transform(pop_dists[region].begin(), pop_dists[region].end(), pop_dists[region].begin(),
                       [&persons_per_agent](auto& c) {
                           return c * persons_per_agent;
                       });
        while (num_agents > 0) {
            for (Eigen::Index i = 0; i < metaregions.rows(); i += 2) {
                for (Eigen::Index j = 0; j < metaregions.cols(); j += 2) {
                    if (metaregions(i, j) == int(region) + 1) {
                        auto status = mio::DiscreteDistribution<int>::get_instance()(pop_dists[region]);
                        agents.push_back({{i, j}, static_cast<mio::mpm::paper::InfectionState>(status), int(region)});
                        pop_dists[region][status] = std::max(0., pop_dists[region][status] - 1);
                        num_agents -= 1;
                    }
                }
            }
        }
    }

    if (save_initialization) {
        std::string save_path = "init" + std::to_string(agents.size()) + ".json";
        Json::Value all_agents;
        for (int i = 0; i < agents.size(); ++i) {
            BOOST_OUTCOME_TRY(agent, mio::serialize_json(agents[i]));
            all_agents[std::to_string(i)] = agent;
        }
        auto write_status = mio::write_json(save_path, all_agents);
    }

    return mio::success(agents);
}

int main()
{
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    std::vector<int> regions        = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    double t_Exposed                = 4.2;
    double t_Carrier                = 4.2;
    double t_Infected               = 7.5;
    double mu_C_R                   = 0.23;
    std::vector<std::vector<double>> pop_dists =
        set_confirmed_case_data(regions, populations, mio::Date(2020, 12, 12), t_Exposed, t_Carrier, t_Infected, mu_C_R)
            .value();

    Eigen::MatrixXi metaregions;

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
    auto agents = create_agents(pop_dists, populations, 100, metaregions, true);

    return 0;
    ;
}
