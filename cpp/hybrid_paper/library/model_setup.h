#ifndef MODEL_SETUP_H_
#define MODEL_SETUP_H_

#include "hybrid_paper/library/weighted_gradient.h"
#include "hybrid_paper/library/infection_state.h"
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
                model.populations[{static_cast<mio::mpm::Region>(k), static_cast<Status>(i)}] = pop_dists[k][i];
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
    double transmission_rate;
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
    std::vector<TransitionRate<Status>> transition_rates;
    std::vector<std::vector<double>> pop_dists;

    ModelSetup(double t_Exposed, double t_Carrier, double t_Infected, double transmission_rate, double mu_C_R,
               double mu_I_D, const Date start_date, const std::vector<int>& region_ids,
               const std::vector<double>& populations, double persons_per_agent,
               Eigen::Ref<const Eigen::MatrixXi> metaregions, const double tmax, double dt,
               Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> commute_weights,
               const WeightedGradient& wg, const std::vector<double>& sigmas, double contact_radius,
               const std::map<std::tuple<Region, Region>, double>& transition_factors)
        : t_Exposed(t_Exposed)
        , t_Carrier(t_Carrier)
        , t_Infected(t_Infected)
        , transmission_rate(transmission_rate)
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

        pop_dists = set_confirmed_case_data(confirmed_cases, region_ids, populations, start_date, t_Exposed, t_Carrier,
                                            t_Infected, mu_C_R)
                        .value();

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
        std::vector<mio::mpm::TransitionRate<Status>> transition_rates;
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