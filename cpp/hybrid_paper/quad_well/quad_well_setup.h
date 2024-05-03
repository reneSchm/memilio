#ifndef QUAD_WELL_SETUP_H
#define QUAD_WELL_SETUP_H

#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/quad_well.h"
#include "memilio/utils/index.h"
#include "models/mpm/model.h"
#include "memilio/math/eigen.h"
#include "memilio/utils/custom_index_array.h"
#include "mpm/region.h"
#include <map>
#include <cstddef>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

template <class Agent>
struct QuadWellSetup {

    using Status = mio::mpm::paper::InfectionState;

    template <class ABM>
    ABM create_abm() const
    {
        return ABM(agents, adoption_rates, contact_radius, sigma,
                   {Status::D}); //{Status::S, Status::E, Status::C, Status::I, Status::R, Status::D}
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

    template <class Sim>
    void redraw_agents_status(Sim& sim) const
    {
        using Index = mio::Index<mio::mpm::Region, Status>;
        // result last value in result timeseries
        sim.get_result().get_last_value() = Eigen::VectorXd::Zero(sim.get_result().get_num_elements());
        auto& sta_rng                     = mio::DiscreteDistribution<int>::get_instance();
        for (auto& agent : sim.get_model().populations) {
            size_t region = qw::well_index(agent.position);
            agent.status  = static_cast<Status>(sta_rng(init_dists[region]));
            auto index    = mio::flatten_index(Index{static_cast<mio::mpm::Region>(region), agent.status},
                                               Index{mio::mpm::Region(4), Status::Count});
            sim.get_result().get_last_value()[index] += 1;
        }
    }

    template <class Sim>
    void redraw_pdmm_populations(Sim& sim) const
    {
        using Index                       = mio::Index<mio::mpm::Region, Status>;
        sim.get_result().get_last_value() = Eigen::VectorXd::Zero(sim.get_result().get_num_elements());
        auto& sta_rng                     = mio::DiscreteDistribution<int>::get_instance();
        for (size_t region = 0; region < 4; ++region) {
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
                                                Index{mio::mpm::Region(4), Status::Count});
                sim.get_result().get_last_value()[index] += 1;
            }
        }
    }

    template <class Model>
    void dummy(Model& /*model*/)
    {
    }

    void read_positions(std::string filename, std::vector<Agent>& agents)
    {
        size_t iter = 0;
        std::string line;
        std::ifstream file(filename);
        if (file.is_open()) {
            while (std::getline(file, line)) {
                if (iter == agents.size()) {
                    break;
                }
                std::stringstream ss(line);
                std::string pos;
                size_t pos_iter = 0;
                while (std::getline(ss, pos, ' ')) {
                    agents[iter].position[pos_iter] = std::stod(pos);
                    ++pos_iter;
                }
                ++iter;
            }
            file.close();
        }
    }

    qw::MetaregionSampler pos_rng{{-2, -2}, {0, 0}, {2, 2}, 0.5};
    // qw::PositionSampler focus_pos_rng{{{0.0001, 0.25, 0.12505}, {0.008, 1.78, 0.93}, {0.1, 0.3, 0.2}}, //0.52
    //                                   {{0.15, 1.65, 0.9}, {0.0001, 0.25, 0.12505}, {0.1, 0.3, 0.2}}};

    qw::PositionSampler focus_pos_rng{{{0.0001, 0.199, 0.1}, {0.008, 1.78, 0.93}, {0.1, 0.3, 0.2}}, //0.52
                                      {{0.15, 1.65, 0.9}, {0.0001, 0.199, 0.1}, {0.1, 0.3, 0.2}}};

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
    const std::vector<std::vector<double>> init_dists;
    //ABM
    double sigma;
    double contact_radius;
    //PDMM
    std::vector<mio::mpm::TransitionRate<Status>> transition_rates;

    QuadWellSetup(double t_Exposed, double t_Carrier, double t_Infected, std::vector<double> transmission_rates,
                  double mu_C_R, double mu_I_D, double tmax, double dt, double sigma, double contact_radius,
                  size_t num_agents, const std::vector<std::vector<double>>& init_dists,
                  const std::map<std::tuple<Status, mio::mpm::Region, mio::mpm::Region>, double>& transition_factors)
        : t_Exposed(t_Exposed)
        , t_Carrier(t_Carrier)
        , t_Infected(t_Infected)
        , mu_C_R(mu_C_R)
        , mu_I_D(mu_I_D)
        , agents(num_agents)
        , init_dists(init_dists)
        , tmax(tmax)
        , dt(dt)
        , num_agents(num_agents)
        , sigma(sigma)
        , contact_radius(contact_radius)
    {
        //initialize agents
        size_t counter = 0;
        auto& sta_rng  = mio::DiscreteDistribution<int>::get_instance();
        bool read_pos  = true;
        if (read_pos) {
            read_positions(mio::base_dir() + "cpp/hybrid_paper/quad_well/input/positions_8000.txt", agents);
        }
        for (auto& agent : agents) {
            agent.status = static_cast<Status>(sta_rng(init_dists[counter]));
            if (!read_pos) {
                agent.position = pos_rng(counter);
            }
            counter = (counter + 1) % 4;
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
            transition_rates.push_back(
                {std::get<0>(rate.first), std::get<1>(rate.first), std::get<2>(rate.first), rate.second});
            transition_rates.push_back(
                {std::get<0>(rate.first), std::get<2>(rate.first), std::get<1>(rate.first), rate.second});
        }
    }

    QuadWellSetup(size_t num_agents)
        : QuadWellSetup(3.0, //t_Exposed
                        3.0, //t_Carrier
                        5.0, //t_Infected
                        std::vector<double>{0.1, 0.3, 0.1, 0.1}, //transmission_rates
                        0.1, //mu_C_R
                        0.004, //mu_I_D
                        150.0, //tmax
                        0.1, //dt
                        0.55, //sigma
                        0.4, //contact_radius
                        num_agents,
                        {{0.99, 0.002, 0.003, 0.005, 0.0, 0.0},
                         {0.99, 0.002, 0.003, 0.005, 0.0, 0.0},
                         {0.99, 0.002, 0.003, 0.005, 0.0, 0.0},
                         {0.99, 0.002, 0.003, 0.005, 0.0, 0.0}}, //pop dists,
                        std::map<std::tuple<Status, mio::mpm::Region, mio::mpm::Region>, double>{
                            //transition rates for sigma = 0.55
                            {{Status::S, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0048}, //0->1
                            {{Status::E, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0048},
                            {{Status::C, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0048},
                            {{Status::I, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0048},
                            {{Status::R, mio::mpm::Region(0), mio::mpm::Region(1)}, 0.0048},
                            {{Status::S, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0048}, //0->2
                            {{Status::E, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0048},
                            {{Status::C, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0048},
                            {{Status::I, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0048},
                            {{Status::R, mio::mpm::Region(0), mio::mpm::Region(2)}, 0.0048},
                            {{Status::S, mio::mpm::Region(0), mio::mpm::Region(3)}, 1e-07}, //0->3 1e-07
                            {{Status::S, mio::mpm::Region(1), mio::mpm::Region(2)}, 1e-07}, //1->2
                            {{Status::S, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0048}, //1->3
                            {{Status::E, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0048},
                            {{Status::C, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0048},
                            {{Status::I, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0048},
                            {{Status::R, mio::mpm::Region(1), mio::mpm::Region(3)}, 0.0048},
                            {{Status::S, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0048}, //2->3
                            {{Status::E, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0048},
                            {{Status::C, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0048},
                            {{Status::I, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0048},
                            {{Status::R, mio::mpm::Region(2), mio::mpm::Region(3)}, 0.0048}})
    {
    }
};

//transition rates for sigma=0.5
// {{mio::mpm::Region(0), mio::mpm::Region(1)}, 0.001196},
// {{mio::mpm::Region(0), mio::mpm::Region(2)}, 0.001196},
// {{mio::mpm::Region(0), mio::mpm::Region(3)}, 1e-07},
// {{mio::mpm::Region(1), mio::mpm::Region(3)}, 0.001196},
// {{mio::mpm::Region(2), mio::mpm::Region(3)}, 0.001196}
// {{mio::mpm::Region(1), mio::mpm::Region(2)}, 1e-07},

//transition rates for sigma=0.55
// {{mio::mpm::Region(0), mio::mpm::Region(1)}, 0.00448308125},
// {{mio::mpm::Region(0), mio::mpm::Region(2)}, 0.00448308125},
// {{mio::mpm::Region(0), mio::mpm::Region(3)}, 1e-07},
// {{mio::mpm::Region(1), mio::mpm::Region(2)}, 1e-07},
// {{mio::mpm::Region(1), mio::mpm::Region(3)}, 0.00448308125},
// {{mio::mpm::Region(2), mio::mpm::Region(3)}, 0.00448308125}

// {0.98, 0.002, 0.008, 0.01, 0.0, 0.0},
//                {0.99, 0.002, 0.003, 0.005, 0.0, 0.0},
//                {0.97, 0.01, 0.01, 0.01, 0.0, 0.0},
//                {0.99, 0.002, 0.003, 0.005, 0.0, 0.0}

// {0.99, 0.002, 0.003, 0.005, 0.0, 0.0},
// {0.98, 0.002, 0.008, 0.01, 0.0, 0.0},
// {0.97, 0.01, 0.01, 0.01, 0.0, 0.0},
// {0.98, 0.002, 0.008, 0.01, 0.0, 0.0}

#endif //QUAD_WELL_SETUP_H