#include "hybrid_paper/initialization.h"

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()

mio::IOResult<std::vector<mio::mpm::ABM<PotentialGermany<mio::mpm::paper::InfectionState>>::Agent>>
create_agents(std::vector<std::vector<double>>& pop_dists, std::vector<double>& populations, double persons_per_agent,
              Eigen::MatrixXi& metaregions, bool save_initialization)
{
    std::vector<mio::mpm::ABM<PotentialGermany<mio::mpm::paper::InfectionState>>::Agent> agents;
    for (size_t region = size_t(0); region < pop_dists.size(); ++region) {
        int num_agents = populations[region] / persons_per_agent;
        std::transform(pop_dists[region].begin(), pop_dists[region].end(), pop_dists[region].begin(),
                       [&persons_per_agent](auto& c) {
                           return c / persons_per_agent;
                       });
        while (num_agents > 0) {
            for (Eigen::Index i = 0; i < metaregions.rows(); i += 2) {
                if (num_agents <= 0) {
                    break;
                }
                for (Eigen::Index j = 0; j < metaregions.cols(); j += 2) {
                    if (num_agents <= 0) {
                        break;
                    }
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
    //number inhabitants per region
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    //county ids
    std::vector<int> regions        = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    double t_Exposed                = 4.2;
    double t_Carrier                = 4.2;
    double t_Infected               = 7.5;
    double mu_C_R                   = 0.23;

    std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =  mio::read_confirmed_cases_data("../../data/Germany/cases_all_county_age_ma7.json").value();

    //vector with entry for every region. Entries are vector with population for every infection state according to initialization
    std::vector<std::vector<double>> pop_dists =
        set_confirmed_case_data(confirmed_cases, regions, populations, mio::Date(2020, 12, 12), t_Exposed, t_Carrier, t_Infected, mu_C_R)
            .value();

    //read map with metaregions
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
    //returns vector with agents. The agents' infection state is set according to pop_dists
    auto agents = create_agents(pop_dists, populations, 100, metaregions, true);

    return 0;
}
