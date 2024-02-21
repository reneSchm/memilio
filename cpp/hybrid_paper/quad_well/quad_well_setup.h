#ifndef QUAD_WELL_SETUP_H
#define QUAD_WELL_SETUP_H

#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/quad_well.h"
#include "models/mpm/model.h"
#include "memilio/math/eigen.h"
#include <map>
#include <cstddef>
#include <vector>

const qw::MetaregionSampler pos_rng{{-2, -2}, {0, 0}, {2, 2}, 0.3};

template <class Agent>
struct QuadWellSetup {

    using Status = mio::mpm::paper::InfectionState;

    template <class ABM>
    ABM create_abm() const
    {
        return ABM(agents, adoption_rates, contact_radius, sigma);
    }

    template <class PDMM>
    PDMM create_pdmm() const
    {
        PDMM model;
        model.parameters.template get<mio::mpm::AdoptionRates<Status>>()   = adoption_rates;
        model.parameters.template get<mio::mpm::TransitionRates<Status>>() = transition_rates;
        model.populations.array().setZero();
        for (auto& agent : agents) {
            model.populations[{static_cast<mio::mpm::Region>(qw::well_index(agent.position)),
                               static_cast<Status>(agent.status)}] += 1;
        }
        return model;
    }

    double t_Exposed;
    double t_Carrier;
    double t_Infected;
    double mu_C_R;
    double mu_I_D;
    std::vector<Agent> agents;
    std::vector<mio::mpm::AdoptionRate<Status>> adoption_rates;
    double tmax;
    double dt;
    size_t num_agents;
    //ABM
    double sigma;
    double contact_radius;
    //PDMM
    std::vector<mio::mpm::TransitionRate<Status>> transition_rates;

    QuadWellSetup(double t_Exposed, double t_Carrier, double t_Infected, std::vector<double> transmission_rates,
                  double mu_C_R, double mu_I_D, double tmax, double dt, double sigma, double contact_radius,
                  size_t num_agents, const std::vector<std::vector<double>>& init_dists,
                  const std::map<std::tuple<mio::mpm::Region, mio::mpm::Region>, double>& transition_factors)
        : t_Exposed(t_Exposed)
        , t_Carrier(t_Carrier)
        , t_Infected(t_Infected)
        , mu_C_R(mu_C_R)
        , mu_I_D(mu_I_D)
        , agents(num_agents)
        , tmax(tmax)
        , dt(dt)
        , num_agents(num_agents)
        , sigma(sigma)
        , contact_radius(contact_radius)
    {
        //initialize agents
        size_t counter = 0;
        auto& sta_rng  = mio::DiscreteDistribution<int>::get_instance();
        for (auto& agent : agents) {
            switch (counter % 4) {
            case 0:
                agent.position = pos_rng(0); //upper left quadrant
                agent.status   = static_cast<Status>(sta_rng(init_dists[0]));
                break;
            case 1:
                agent.position = pos_rng(1); //upper right quadrant
                agent.status   = static_cast<Status>(sta_rng(init_dists[1]));
                break;
            case 2:
                agent.position = pos_rng(2);
                ; //lower left quadrant
                agent.status = static_cast<Status>(sta_rng(init_dists[2]));
                break;
            case 3:
                agent.position = pos_rng(3);
                ; //lower right quadrant
                agent.status = static_cast<Status>(sta_rng(init_dists[3]));
                break;
            }
            ++counter;
        }

        //set adoption rates
        for (size_t r = 0; r < 4; ++r) {
            adoption_rates.push_back(
                {Status::S, Status::E, mio::mpm::Region(r), transmission_rates[r], {Status::C, Status::I}, {1, 1}});
            adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(r), 1.0 / t_Exposed});
            adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(r), mu_C_R / t_Carrier});
            adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(r), (1 - mu_C_R) / t_Carrier});
            adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(r), (1 - mu_I_D) / t_Infected});
            adoption_rates.push_back({Status::I, Status::D, mio::mpm::Region(r), mu_I_D / t_Infected});
        }

        //set transition rates
        std::vector<Status> transitioning_states{Status::S, Status::E, Status::C, Status::I, Status::R};
        //No adapted transition behaviour when infected
        for (auto& rate : transition_factors) {
            for (auto s : transitioning_states) {
                transition_rates.push_back({s, std::get<0>(rate.first), std::get<1>(rate.first), rate.second});
                transition_rates.push_back({s, std::get<1>(rate.first), std::get<0>(rate.first), rate.second});
            }
        }
    }

    QuadWellSetup(size_t num_agents)
        : QuadWellSetup(3.0, //t_Exposed
                        3.0, //t_Carrier
                        5.0, //t_Infected
                        std::vector<double>(4, 0.35), //transmission_rates
                        0.1, //mu_C_R
                        0.004, //mu_I_D
                        100.0, //tmax
                        0.1, //dt
                        0.4, //sigma
                        0.4, //contact_radius
                        num_agents,
                        {{0.99, 0.002, 0.003, 0.005, 0.0, 0.0},
                         {0.99, 0.002, 0.003, 0.005, 0.0, 0.0},
                         {0.99, 0.002, 0.003, 0.005, 0.0, 0.0},
                         {0.99, 0.002, 0.003, 0.005, 0.0, 0.0}}, //pop dists
                        std::map<std::tuple<mio::mpm::Region, mio::mpm::Region>, double>{
                            {{mio::mpm::Region(0), mio::mpm::Region(1)}, 0.02},
                            {{mio::mpm::Region(0), mio::mpm::Region(2)}, 0.02},
                            {{mio::mpm::Region(0), mio::mpm::Region(3)}, 0.02},
                            {{mio::mpm::Region(1), mio::mpm::Region(2)}, 0.02},
                            {{mio::mpm::Region(1), mio::mpm::Region(3)}, 0.02},
                            {{mio::mpm::Region(2), mio::mpm::Region(3)}, 0.02}}) //transition rates
    {
    }
};

#endif //QUAD_WELL_SETUP_H