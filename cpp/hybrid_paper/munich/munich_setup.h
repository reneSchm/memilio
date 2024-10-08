#ifndef MODEL_SETUP_H_
#define MODEL_SETUP_H_

#include "hybrid_paper/library/metaregion_sampler.h"
#include "hybrid_paper/library/weighted_gradient.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "memilio/utils/compiler_diagnostics.h"

namespace mio
{
namespace mpm
{
namespace paper
{

template <class Agent>
struct MunichSetup {
    using Status = mio::mpm::paper::InfectionState;

    template <class ABM>
    ABM create_abm() const
    {
        return ABM({commute_weights, metaregions, {metaregions}}, agents, adoption_rates, wg.gradient, metaregions,
                   {Status::D}, sigmas, contact_radius);
    }

    template <class PDMM>
    PDMM create_pdmm() const
    {
        PDMM model;
        model.populations.array().setZero();
        for (auto& agent : agents) {
            model.populations[{static_cast<mio::mpm::Region>(agent.region), agent.status}] += 1;
        }
        // for (size_t k = 0; k < region_ids.size(); ++k) {
        //     for (size_t i = 0; i < static_cast<size_t>(Status::Count); ++i) {
        //         model.populations[{static_cast<mio::mpm::Region>(k), static_cast<Status>(i)}] = pop_dists_scaled[k][i];
        //     }
        // }
        model.parameters.template get<TransitionRates<Status>>() = transition_rates;
        model.parameters.template get<AdoptionRates<Status>>()   = adoption_rates;
        return model;
    }

    template <class ABM>
    void draw_ABM_population(ABM& model) const
    {
        model.populations.clear();
        std::vector<std::vector<double>> pop_dists;
        std::copy(pop_dists_scaled.begin(), pop_dists_scaled.end(), std::back_inserter(pop_dists));
        for (size_t region = 0; region < region_ids.size(); ++region) {
            std::transform(pop_dists[region].begin(), pop_dists[region].end(), pop_dists[region].begin(),
                           [this](auto& c) {
                               return c * persons_per_agent;
                           });
        }
        model.populations =
            create_agents<Agent>(pop_dists, populations, persons_per_agent, metaregion_sampler, false).value();
    }

    template <class Sim>
    void redraw_agents_status(Sim& sim) const
    {
        using Index = mio::Index<mio::mpm::Region, Status>;
        // result last value in result timeseries
        sim.get_result().get_last_value() = Eigen::VectorXd::Zero(sim.get_result().get_num_elements());
        auto& sta_rng                     = mio::DiscreteDistribution<int>::get_instance();
        for (auto& agent : sim.get_model().populations) {
            agent.status = static_cast<Status>(sta_rng(init_dists[agent.region]));
            auto index   = mio::flatten_index(Index{static_cast<mio::mpm::Region>(agent.region), agent.status},
                                              Index{mio::mpm::Region(8), Status::Count});
            sim.get_result().get_last_value()[index] += 1;
        }
    }

    template <class Sim>
    void redraw_pdmm_populations(Sim& sim) const
    {
        using Index                       = mio::Index<mio::mpm::Region, Status>;
        sim.get_result().get_last_value() = Eigen::VectorXd::Zero(sim.get_result().get_num_elements());
        auto& sta_rng                     = mio::DiscreteDistribution<int>::get_instance();
        for (size_t region = 0; region < 8; ++region) {
            double local_pop = 0;
            for (size_t status = 0; status < static_cast<size_t>(Status::Count); ++status) {
                local_pop +=
                    sim.get_model().populations[{static_cast<mio::mpm::Region>(region), static_cast<Status>(status)}];
                sim.get_model().populations[{static_cast<mio::mpm::Region>(region), static_cast<Status>(status)}] = 0;
            }
            for (size_t agent = 0; agent < local_pop; ++agent) {
                auto new_status = static_cast<Status>(sta_rng(init_dists[region]));
                sim.get_model().populations[{static_cast<mio::mpm::Region>(region), new_status}] += 1;
                auto index = mio::flatten_index(Index{static_cast<mio::mpm::Region>(region), new_status},
                                                Index{mio::mpm::Region(8), Status::Count});
                sim.get_result().get_last_value()[index] += 1;
            }
        }
    }

    template <class Model>
    void dummy(Model& /*model*/)
    {
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
    const std::vector<std::vector<double>> init_dists;
    //Model requirements
    std::vector<Agent> agents;
    Eigen::MatrixXi metaregions;
    std::vector<AdoptionRate<Status>> adoption_rates;
    double tmax;
    double dt;
    //ABM requirements
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights;
    MetaregionSampler metaregion_sampler;
    WeightedGradient wg;
    std::vector<double> sigmas;
    double contact_radius;
    //PDMM requirements
    std::vector<TransitionRate<Status>> commute_rates;
    std::vector<TransitionRate<Status>> transition_rates;
    std::vector<std::vector<double>> pop_dists_scaled;

    MunichSetup(double t_E, double t_C, double t_I, std::vector<double> transm_rats, double m_C_R, double m_I_D,
                Date date, const std::vector<int>& regn_ids, const std::vector<double>& pops, double ppa,
                const std::vector<std::vector<double>> init, Eigen::Ref<const Eigen::MatrixXi> metaregns,
                const double t_max, double delta_t,
                Eigen::Ref<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> comm_weights,
                const std::vector<double>& sgms, double cr)
        : t_Exposed(t_E)
        , t_Carrier(t_C)
        , t_Infected(t_I)
        , transmission_rates(transm_rats)
        , mu_C_R(m_C_R)
        , mu_I_D(m_I_D)
        , start_date(date)
        , region_ids(regn_ids)
        , populations(pops)
        , persons_per_agent(ppa)
        , init_dists(init)
        , metaregions(metaregns)
        , tmax(t_max)
        , dt(delta_t)
        , commute_weights(comm_weights)
        , metaregion_sampler(metaregns)
        , wg(mio::base_dir() + "potentially_germany_grad.json", mio::base_dir() + "boundary_ids.pgm")
        , sigmas(sgms)
        , contact_radius(cr)

    {
        // //create agents
        // std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =
        //     mio::read_confirmed_cases_data(mio::base_dir() + "data/Germany/cases_all_county_age_ma7.json").value();
        // std::sort(confirmed_cases.begin(), confirmed_cases.end(), [](auto&& a, auto&& b) {
        //     return std::make_tuple(get_region_id(a), a.date) < std::make_tuple(get_region_id(b), b.date);
        // });

        // std::vector<std::vector<double>> pop_dists =
        //     set_confirmed_case_data(confirmed_cases, region_ids, populations, start_date, t_Exposed, t_Carrier,
        //                             t_Infected, mu_C_R, scaling_factor_infected, set_only_infected)
        //         .value();
        // std::copy(pop_dists.begin(), pop_dists.end(), std::back_inserter(pop_dists_scaled));
        // for (size_t region = 0; region < region_ids.size(); ++region) {
        //     std::transform(pop_dists_scaled[region].begin(), pop_dists_scaled[region].end(),
        //                    pop_dists_scaled[region].begin(), [&persons_per_agent](auto& c) {
        //                        return c / persons_per_agent;
        //                    });
        // }
        // agents = create_agents<Agent>(pop_dists, populations, persons_per_agent, metaregion_sampler, false).value();

        agents = create_susceptible_agents<Agent>(populations, persons_per_agent, metaregion_sampler, false).value();
        std::cout << "num_agents: " << agents.size() << "\n";
        auto& sta_rng     = mio::DiscreteDistribution<int>::get_instance();
        bool has_infected = false; //TODO
        for (auto& a : agents) {
            a.status = static_cast<Status>(sta_rng(init_dists[a.region]));
            if (!has_infected && a.region == 4) {
                a.status     = Status::I;
                has_infected = true;
            }
        }

        //adoption_rates
        for (int i = 0; i < metaregions.maxCoeff(); ++i) {
            adoption_rates.push_back(
                {Status::S, Status::E, mio::mpm::Region(i), transmission_rates[i], {Status::C, Status::I}, {1, 1}});
            adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(i), 1.0 / t_Exposed, {}, {}});
            adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(i), mu_C_R / t_Carrier, {}, {}});
            adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(i), (1 - mu_C_R) / t_Carrier, {}, {}});
            adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), (1 - mu_I_D) / t_Infected, {}, {}});
            adoption_rates.push_back({Status::I, Status::D, mio::mpm::Region(i), mu_I_D / t_Infected, {}, {}});
        }

        std::vector<Status> transitioning_states{Status::S, Status::E, Status::C, Status::I, Status::R};
        //No adapted transition behaviour when infected
        for (size_t from = 0; from < static_cast<size_t>(metaregions.maxCoeff()); ++from) {
            for (size_t to = 0; to < static_cast<size_t>(metaregions.maxCoeff()); ++to) {
                if (from != to) {
                    for (auto s : transitioning_states) {
                        transition_rates.push_back(
                            {s, Region(from), Region(to),
                             (commute_weights(from, to) + commute_weights(to, from)) / populations[from]});
                    }
                }
            }
        }

        const std::vector<double> weights(14, 10);
        wg.apply_weights(weights);
    }

    MunichSetup()
        : MunichSetup(
              3.0, //t_Exposed
              3.0, //t_Carrier
              6.0, //t_Infected
              std::vector<double>(8, 0.2), //transmission rates
              0.2, //mu_C_R
              0.003, //mu_I_D
              mio::Date(2021, 3, 1), //start date (not relevant)
              {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175}, //region ids
              {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562}, //populations
              40, //persons per agent
              {{1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
               {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
               {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
               {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
               {0.998, 0.0005, 0.0005, 0.001, 0.0, 0.0},
               {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
               {1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
               {1.0, 0.0, 0.0, 0.0, 0.0, 0.0}}, //init dists
              []() {
                  const auto fname = mio::base_dir() + "metagermany.pgm";
                  std::ifstream ifile(fname);
                  if (!ifile.is_open()) {
                      mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
                      std::abort();
                  }
                  else {
                      auto _metaregions = mio::mpm::read_pgm_raw(ifile).first;
                      ifile.close();
                      return _metaregions;
                  }
              }(), //metaregions
              50, //tmax
              0.1, //dt
              []() {
                  const std::vector<int> county_ids = {233, 228, 242, 223, 238, 232, 231, 229};
                  Eigen::MatrixXd reference_commuters =
                      get_transition_matrix(mio::base_dir() + "data/mobility/").value();
                  std::vector<double> pops{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
                  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _commute_weights(8, 8);
                  _commute_weights.setZero();
                  for (int i = 0; i < 8; i++) {
                      for (int j = 0; j < 8; j++) {
                          if (i != j) {
                              _commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
                          }
                      }
                      _commute_weights(i, i) = pops[i] - _commute_weights.row(i).sum();
                  }
                  return _commute_weights;
              }(), //commute weights
              std::vector<double>(8, 10), //sigmas
              50 //contact radius
          )
    {
    }
};

} // namespace paper
} // namespace mpm
} // namespace mio

#endif //MODEL_SETUP_H
