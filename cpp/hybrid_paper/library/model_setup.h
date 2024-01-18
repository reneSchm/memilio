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
        , transmission_rates(std::vector<double>{0.6, 0.6, 0.45, 0.27, 0.27, 0.5, 0.45, 0.45})
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
        , tmax(30) //TODO
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
            {{Region(0), Region(1)}, 0.0226967},  {{Region(0), Region(2)}, 0.0314005},
            {{Region(0), Region(3)}, 0.218044},   {{Region(0), Region(4)}, 0.0390458},
            {{Region(0), Region(5)}, 0.00584238}, {{Region(0), Region(6)}, 0.00232734},
            {{Region(0), Region(7)}, 0.00268022}, {{Region(1), Region(0)}, 0.0319142},
            {{Region(1), Region(2)}, 0.00627085}, {{Region(1), Region(3)}, 0.262424},
            {{Region(1), Region(4)}, 0.0445099},  {{Region(1), Region(5)}, 0.0197876},
            {{Region(1), Region(6)}, 0.0061859},  {{Region(1), Region(7)}, 0.00456606},
            {{Region(2), Region(0)}, 0.0501912},  {{Region(2), Region(1)}, 0.00712848},
            {{Region(2), Region(3)}, 0.223801},   {{Region(2), Region(4)}, 0.0480711},
            {{Region(2), Region(5)}, 0.00562728}, {{Region(2), Region(6)}, 0.00402513},
            {{Region(2), Region(7)}, 0.00305067}, {{Region(3), Region(0)}, 0.0320357},
            {{Region(3), Region(1)}, 0.0274204},  {{Region(3), Region(2)}, 0.0205713},
            {{Region(3), Region(4)}, 0.114633},   {{Region(3), Region(5)}, 0.0228892},
            {{Region(3), Region(6)}, 0.0299782},  {{Region(3), Region(7)}, 0.0204995},
            {{Region(4), Region(0)}, 0.0243959},  {{Region(4), Region(1)}, 0.0197778},
            {{Region(4), Region(2)}, 0.0187904},  {{Region(4), Region(3)}, 0.487484},
            {{Region(4), Region(5)}, 0.0379266},  {{Region(4), Region(6)}, 0.0259204},
            {{Region(4), Region(7)}, 0.0400258},  {{Region(5), Region(0)}, 0.00704976},
            {{Region(5), Region(1)}, 0.0169807},  {{Region(5), Region(2)}, 0.00424808},
            {{Region(5), Region(3)}, 0.187986},   {{Region(5), Region(4)}, 0.0732462},
            {{Region(5), Region(6)}, 0.0743232},  {{Region(5), Region(7)}, 0.00918042},
            {{Region(6), Region(0)}, 0.00364346}, {{Region(6), Region(1)}, 0.00688711},
            {{Region(6), Region(2)}, 0.00394225}, {{Region(6), Region(3)}, 0.319425},
            {{Region(6), Region(4)}, 0.0649461},  {{Region(6), Region(5)}, 0.096426},
            {{Region(6), Region(7)}, 0.0422341},  {{Region(7), Region(0)}, 0.00405251},
            {{Region(7), Region(1)}, 0.00490993}, {{Region(7), Region(2)}, 0.00288575},
            {{Region(7), Region(3)}, 0.210963},   {{Region(7), Region(4)}, 0.0968616},
            {{Region(7), Region(5)}, 0.0115036},  {{Region(7), Region(6)}, 0.0407908}};
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