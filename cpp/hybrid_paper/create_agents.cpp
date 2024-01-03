#include "initialization.h"
#include "infection_state.h"
#include "mpm/abm.h"
#include "mpm/potentials/potential_germany.h"
#include "mpm/potentials/map_reader.h"
#include "hybrid_paper/metaregion_sampler.h"

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()

std::vector<std::vector<std::pair<size_t, size_t>>> get_region_indices(Eigen::MatrixXi& metaregions)
{
    std::vector<std::vector<std::pair<size_t, size_t>>> indices(metaregions.maxCoeff());
    for (Eigen::Index i = 0; i < metaregions.rows(); ++i) {
        for (Eigen::Index j = 0; j < metaregions.cols(); ++j) {
            if (metaregions(i, j) != 0) {
                indices[metaregions(i, j) - 1].push_back({i, j});
            }
        }
    }
    return indices;
}

mio::IOResult<std::vector<mio::mpm::ABM<GradientGermany<mio::mpm::paper::InfectionState>>::Agent>>
create_susceptible_agents(std::vector<double>& populations, double persons_per_agent, Eigen::MatrixXi& metaregions,
                          MetaregionSampler& metaregion_sampler, bool save_initialization)
{
    std::vector<mio::mpm::ABM<GradientGermany<mio::mpm::paper::InfectionState>>::Agent> agents;
    std::vector<std::vector<std::pair<size_t, size_t>>> indices = get_region_indices(metaregions);
    for (size_t region = size_t(0); region < indices.size(); ++region) {
        std::vector<std::pair<size_t, size_t>>& region_indices = indices[region];
        int num_agents                                         = populations[region] / persons_per_agent;
        while (num_agents > 0) {
            Eigen::Vector2d position = metaregion_sampler(region);
            //mio::DiscreteDistribution<int>::get_instance()(std::vector<double>(region_indices.size(), 1));
            agents.push_back({position, mio::mpm::paper::InfectionState::S, int(region)});
            num_agents -= 1;
        }
    }
    if (save_initialization) {
        std::string save_path = "initSusceptible" + std::to_string(agents.size()) + ".json";
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
    std::vector<int> regions = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    double t_Exposed         = 4.2;
    double t_Carrier         = 4.2;
    double t_Infected        = 7.5;
    double mu_C_R            = 0.23;

    std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =
        mio::read_confirmed_cases_data("../../data/Germany/cases_all_county_age_ma7.json").value();

    //vector with entry for every region. Entries are vector with population for every infection state according to initialization
    std::vector<std::vector<double>> pop_dists =
        set_confirmed_case_data(confirmed_cases, regions, populations, mio::Date(2021, 3, 1), t_Exposed, t_Carrier,
                                t_Infected, mu_C_R)
            .value();

    //read map with metaregions
    Eigen::MatrixXi metaregions;

    //std::cerr << "Setup: Read metaregions.\n" << std::flush;
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

    MetaregionSampler metaregion_sampler(metaregions);

    //returns vector with agents. The agents' infection state is set according to pop_dists
    //auto agents = create_agents(pop_dists, populations, 1000, metaregions, true).value();
    auto agents = create_susceptible_agents(populations, 100000, metaregions, metaregion_sampler, true).value();

    //for density plot
    // std::vector<mio::mpm::ABM<GradientGermany<mio::mpm::paper::InfectionState>>::Agent> agents;
    // read_initialization("initSusceptible9375.json", agents);

    // for (auto& a : agents) {
    //     std::cout << a.position[0] << " " << a.position[1] << " ";
    // }

    return 0;
}
