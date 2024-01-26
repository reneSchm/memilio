#ifndef MODEL_SETUP_H_
#define MODEL_SETUP_H_

#include "hybrid_paper/library/weighted_gradient.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/initialization.h"
#include "potentials/commuting_potential.h"
namespace mio
{
namespace mpm
{
namespace paper
{

template <class Agent>
struct ModelSetup {
    using Status = mio::mpm::paper::InfectionState;

    template <class ABM>
    ABM create_abm() const
    {
        return ABM(k_provider, agents, adoption_rates, wg.gradient, metaregions, {Status::D}, sigmas, contact_radius);
    }

    template <class PDMM>
    PDMM create_pdmm() const
    {
        PDMM model;
        for (size_t k = 0; k < region_ids.size(); ++k) {
            for (int i = 0; i < static_cast<size_t>(Status::Count); ++i) {
                model.populations[{static_cast<mio::mpm::Region>(k), static_cast<Status>(i)}] = pop_dists_scaled[k][i];
            }
        }
        model.parameters.template get<TransitionRates<Status>>() = transition_rates;
        model.parameters.template get<AdoptionRates<Status>>()   = adoption_rates;
        return model;
    }

    //parameters
    double t_Exposed;
    double t_Carrier;
    double t_Infected;
    std::vector<double> transmission_rates;
    double mu_C_R;
    double mu_I_D;
    Date start_date;
    std::vector<int> region_ids;
    std::vector<double> populations;
    double persons_per_agent;
    //Model requirements
    std::vector<Agent> agents;
    Eigen::MatrixXi metaregions;
    std::vector<AdoptionRate<Status>> adoption_rates;
    double tmax;
    double dt;
    //ABM requirements
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights;
    StochastiK k_provider;
    WeightedGradient wg;
    std::vector<double> sigmas;
    double contact_radius;
    //PDMM requirements
    std::vector<TransitionRate<Status>> commute_rates;
    std::vector<TransitionRate<Status>> transition_rates;
    std::vector<std::vector<double>> pop_dists_scaled;

    ModelSetup(double t_Exposed, double t_Carrier, double t_Infected, double transmission_rate, double mu_C_R,
               double mu_I_D, const Date start_date, const std::vector<int>& region_ids,
               const std::vector<double>& populations, double persons_per_agent,
               Eigen::Ref<const Eigen::MatrixXi> metaregions, const double tmax, double dt,
               Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> commute_weights,
               const WeightedGradient& wg, const std::vector<double>& sigmas, double contact_radius,
               const std::map<std::tuple<Region, Region>, double>& transition_factors,
               double scaling_factor_infected = 1.0, bool set_only_infected = false)
        : t_Exposed(t_Exposed)
        , t_Carrier(t_Carrier)
        , t_Infected(t_Infected)
        , transmission_rates(std::vector<double>(8, transmission_rate))
        , mu_C_R(mu_C_R)
        , mu_I_D(mu_I_D)
        , start_date(start_date)
        , region_ids(region_ids)
        , populations(populations)
        , persons_per_agent(persons_per_agent)
        , metaregions(metaregions)
        , tmax(tmax)
        , dt(dt)
        , commute_weights(commute_weights)
        , k_provider(commute_weights, metaregions, {metaregions})
        , wg(wg)
        , sigmas(sigmas)
        , contact_radius(contact_radius)

    {
        //create agents
        std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =
            mio::read_confirmed_cases_data(mio::base_dir() + "data/Germany/cases_all_county_age_ma7.json").value();
        std::sort(confirmed_cases.begin(), confirmed_cases.end(), [](auto&& a, auto&& b) {
            return std::make_tuple(get_region_id(a), a.date) < std::make_tuple(get_region_id(b), b.date);
        });

        std::vector<std::vector<double>> pop_dists =
            set_confirmed_case_data(confirmed_cases, region_ids, populations, start_date, t_Exposed, t_Carrier,
                                    t_Infected, mu_C_R, scaling_factor_infected, set_only_infected)
                .value();
        std::copy(pop_dists.begin(), pop_dists.end(), std::back_inserter(pop_dists_scaled));
        for (size_t region = 0; region < region_ids.size(); ++region) {
            std::transform(pop_dists_scaled[region].begin(), pop_dists_scaled[region].end(),
                           pop_dists_scaled[region].begin(), [&persons_per_agent](auto& c) {
                               return c / persons_per_agent;
                           });
        }
        agents = create_agents(pop_dists, populations, persons_per_agent, {metaregions}, false).value();
        //adoption_rates
        for (int i = 0; i < metaregions.maxCoeff(); ++i) {
            adoption_rates.push_back(
                {Status::S, Status::E, mio::mpm::Region(i), transmission_rate, {Status::C, Status::I}, {1, 1}});
            adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(i), 1.0 / t_Exposed});
            adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(i), mu_C_R / t_Carrier});
            adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(i), (1 - mu_C_R) / t_Carrier});
            adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), (1 - mu_I_D) / t_Infected});
            adoption_rates.push_back({Status::I, Status::D, mio::mpm::Region(i), mu_I_D / t_Infected});
        }

        std::vector<Status> transitioning_states{Status::S, Status::E, Status::C, Status::I, Status::R};
        //No adapted transition behaviour when infected
        for (auto& rate : transition_factors) {
            for (auto s : transitioning_states) {
                transition_rates.push_back({s, std::get<0>(rate.first), std::get<1>(rate.first), rate.second});
            }
        }
    }

    ModelSetup()
        : t_Exposed(3.67652)
        , t_Carrier(2.71414)
        , t_Infected(5)
        , transmission_rates(std::vector<double>{0.37, 0.4, 0.35, 0.25, 0.34, 0.38, 0.35, 0.35})
        , mu_C_R(0.1)
        , mu_I_D(0.004)
        , start_date(mio::Date(2021, 3, 1))
        , region_ids({9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175})
        , populations({218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562})
        , persons_per_agent(100)
        , metaregions([]() {
            const auto fname = mio::base_dir() + "metagermany.pgm";
            std::ifstream ifile(fname);
            if (!ifile.is_open()) {
                mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
                std::abort();
            }
            else {
                auto metaregions = mio::mpm::read_pgm_raw(ifile).first;
                ifile.close();
                return metaregions;
            }
        }())
        , tmax(30)
        , dt(0.1)
        , commute_weights([]() {
            std::vector<double> populations     = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
            const std::vector<int> county_ids   = {233, 228, 242, 223, 238, 232, 231, 229};
            Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();
            Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(8, 8);
            commute_weights.setZero();
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    if (i != j) {
                        commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
                    }
                }
                commute_weights(i, i) = populations[i] - commute_weights.row(i).sum();
            }
            return commute_weights;
        }())
        , k_provider(commute_weights, metaregions, {metaregions})
        , wg(mio::base_dir() + "potentially_germany_grad.json", mio::base_dir() + "boundary_ids.pgm")
        , sigmas(8, 10)
        , contact_radius(50)
    {

        const std::vector<double> weights(14, 500);
        wg.apply_weights(weights);

        std::map<std::tuple<Region, Region>, double> transition_factors{
            {{Region(0), Region(1)}, 0.022904},   {{Region(0), Region(2)}, 0.03138},
            {{Region(0), Region(3)}, 0.218554},   {{Region(0), Region(4)}, 0.0382919},
            {{Region(0), Region(5)}, 0.00583277}, {{Region(0), Region(6)}, 0.00234107},
            {{Region(0), Region(7)}, 0.00272827}, {{Region(1), Region(0)}, 0.0322057},
            {{Region(1), Region(2)}, 0.00636546}, {{Region(1), Region(3)}, 0.262704},
            {{Region(1), Region(4)}, 0.044089},   {{Region(1), Region(5)}, 0.0201274},
            {{Region(1), Region(6)}, 0.00648516}, {{Region(1), Region(7)}, 0.00469156},
            {{Region(2), Region(0)}, 0.0501583},  {{Region(2), Region(1)}, 0.00723602},
            {{Region(2), Region(3)}, 0.224284},   {{Region(2), Region(4)}, 0.0477814},
            {{Region(2), Region(5)}, 0.00560753}, {{Region(2), Region(6)}, 0.00393295},
            {{Region(2), Region(7)}, 0.00307481}, {{Region(3), Region(0)}, 0.0321107},
            {{Region(3), Region(1)}, 0.0274496},  {{Region(3), Region(2)}, 0.0206157},
            {{Region(3), Region(4)}, 0.114467},   {{Region(3), Region(5)}, 0.0228919},
            {{Region(3), Region(6)}, 0.0299915},  {{Region(3), Region(7)}, 0.0204791},
            {{Region(4), Region(0)}, 0.0239249},  {{Region(4), Region(1)}, 0.0195908},
            {{Region(4), Region(2)}, 0.0186772},  {{Region(4), Region(3)}, 0.486778},
            {{Region(4), Region(5)}, 0.0380595},  {{Region(4), Region(6)}, 0.0257428},
            {{Region(4), Region(7)}, 0.0400996},  {{Region(5), Region(0)}, 0.00703816},
            {{Region(5), Region(1)}, 0.0172723},  {{Region(5), Region(2)}, 0.00423317},
            {{Region(5), Region(3)}, 0.188007},   {{Region(5), Region(4)}, 0.073503},
            {{Region(5), Region(6)}, 0.0742221},  {{Region(5), Region(7)}, 0.00894681},
            {{Region(6), Region(0)}, 0.00366496}, {{Region(6), Region(1)}, 0.00722029},
            {{Region(6), Region(2)}, 0.00385197}, {{Region(6), Region(3)}, 0.319567},
            {{Region(6), Region(4)}, 0.0645011},  {{Region(6), Region(5)}, 0.0962949},
            {{Region(6), Region(7)}, 0.0425458},  {{Region(7), Region(0)}, 0.00412517},
            {{Region(7), Region(1)}, 0.00504488}, {{Region(7), Region(2)}, 0.00290859},
            {{Region(7), Region(3)}, 0.210753},   {{Region(7), Region(4)}, 0.0970402},
            {{Region(7), Region(5)}, 0.0112108},  {{Region(7), Region(6)}, 0.0410919}};
        double scaling_factor_infected = 1.0;
        bool set_only_infected         = false;
        //create agents
        std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =
            mio::read_confirmed_cases_data(mio::base_dir() + "data/Germany/cases_all_county_age_ma7.json").value();
        std::sort(confirmed_cases.begin(), confirmed_cases.end(), [](auto&& a, auto&& b) {
            return std::make_tuple(get_region_id(a), a.date) < std::make_tuple(get_region_id(b), b.date);
        });

        std::vector<std::vector<double>> pop_dists =
            set_confirmed_case_data(confirmed_cases, region_ids, populations, start_date, t_Exposed, t_Carrier,
                                    t_Infected, mu_C_R, scaling_factor_infected, set_only_infected)
                .value();
        std::copy(pop_dists.begin(), pop_dists.end(), std::back_inserter(pop_dists_scaled));
        for (size_t region = 0; region < region_ids.size(); ++region) {
            std::transform(pop_dists_scaled[region].begin(), pop_dists_scaled[region].end(),
                           pop_dists_scaled[region].begin(), [this](auto& c) {
                               return c / persons_per_agent;
                           });
        }
        agents = create_agents(pop_dists, populations, persons_per_agent, {metaregions}, false).value();
        //adoption_rates
        for (int i = 0; i < metaregions.maxCoeff(); ++i) {
            adoption_rates.push_back(
                {Status::S, Status::E, mio::mpm::Region(i), transmission_rates[i], {Status::C, Status::I}, {1, 1}});
            adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(i), 1.0 / t_Exposed});
            adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(i), mu_C_R / t_Carrier});
            adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(i), (1 - mu_C_R) / t_Carrier});
            adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), (1 - mu_I_D) / t_Infected});
            adoption_rates.push_back({Status::I, Status::D, mio::mpm::Region(i), mu_I_D / t_Infected});
        }

        std::vector<Status> transitioning_states{Status::S, Status::E, Status::C, Status::I, Status::R};
        //No adapted transition behaviour when infected
        for (auto& rate : transition_factors) {
            for (auto s : transitioning_states) {
                transition_rates.push_back({s, std::get<0>(rate.first), std::get<1>(rate.first), rate.second});
            }
        }
    }
};

} // namespace paper
} // namespace mpm
} // namespace mio

#endif //MODEL_SETUP_H