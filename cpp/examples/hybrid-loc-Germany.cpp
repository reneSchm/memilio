#include "memilio/compartments/simulation.h"
#include "memilio/config.h"
#include "memilio/utils/logging.h"
#include "memilio/io/json_serializer.h"
#include "memilio/utils/time_series.h"
#include "mpm/abm.h"
#include "mpm/potentials/potential_germany.h"
#include "mpm/model.h"
#include "mpm/region.h"
#include "mpm/smm.h"
#include "mpm/pdmm.h"
#include "mpm/utility.h"

#include <algorithm>
#include <cstdio>
#include <list>
#include <map>
#include <chrono>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()
#define PRECISION 17

#define restart_timer(timer, description)                                                                              \
    {                                                                                                                  \
        TIME_TYPE new_time = TIME_NOW;                                                                                 \
        std::cout << "\r" << description << " :: " << PRINTABLE_TIME(new_time - timer) << std::endl << std::flush;     \
        timer = new_time;                                                                                              \
    }

//#undef restart_timer(timer, description)
//#define restart_timer(timer, description)

enum class InfectionState
{
    S,
    E,
    C,
    I,
    R,
    D,
    Count
};

template <class Status, class Agent>
mio::IOResult<void> create_start_initialization(std::vector<Agent>& agents, std::vector<double>& pop_dist,
                                                Eigen::MatrixXd& potential, Eigen::MatrixXi& metaregions)
{
    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    auto& sta_rng = mio::DiscreteDistribution<int>::get_instance();
    for (auto& a : agents) {
        Eigen::Vector2d pos_candidate{pos_rng(0.0, double(potential.rows())), pos_rng(0.0, double(potential.cols()))};
        while (metaregions(pos_candidate[0], pos_candidate[1]) == 0 ||
               potential(pos_candidate[0], pos_candidate[1]) != 0) {
            pos_candidate = {pos_rng(0.0, double(potential.rows())), pos_rng(0.0, double(potential.cols()))};
        }
        a.position = {pos_candidate[0], pos_candidate[1]};
        a.land     = metaregions(pos_candidate[0], pos_candidate[1]) - 1;
        //only infected agents in focus region
        if (a.land == 8) {
            a.status = static_cast<Status>(sta_rng(pop_dist));
        }
        else {
            a.status = Status::S;
        }
    }

    Json::Value all_agents;
    for (int i = 0; i < agents.size(); ++i) {
        BOOST_OUTCOME_TRY(agent, mio::serialize_json(agents[i]));
        all_agents[std::to_string(i)] = agent;
    }
    auto write_status = mio::write_json("initialization.json", all_agents);
}

template <class Agent>
void read_initialization(std::string filename, std::vector<Agent>& agents, int n_agents)
{
    auto result = mio::read_json(filename).value();
    for (int i = 0; i < n_agents; ++i) {
        auto& a = mio::deserialize_json(result[std::to_string(i)], mio::Tag<Agent>{}).value();
        agents.push_back(Agent{a.position, a.status, a.land});
    }
}

int main(int argc, char** argv)
{
    using namespace mio::mpm;
    using Status = ABM<PotentialGermany<InfectionState>>::Status;

    Eigen::MatrixXd potential;
    Eigen::MatrixXi metaregions;

    std::cerr << "Setup: Read potential.\n" << std::flush;
    {
        const auto fname = "~input/potentially_germany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            return 1;
        }
        else {
            potential = read_pgm(ifile);
            ifile.close();
        }
    }
    std::cerr << "Setup: Read metaregions.\n" << std::flush;
    {
        const auto fname = "~input/metagermany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            return 1;
        }
        else {
            metaregions = (16 * read_pgm(ifile))
                              .unaryExpr([](double c) {
                                  return std::round(c);
                              })
                              .cast<int>();
            ifile.close();
        }
    }

    //read agents
    std::vector<ABM<PotentialGermany<InfectionState>>::Agent> agents;
    read_initialization<ABM<PotentialGermany<InfectionState>>::Agent>("~/initialization.json", agents, 16 * 100);

    std::vector<ABM<PotentialGermany<InfectionState>>::Agent> agents_focus_region;
    std::copy_if(agents.begin(), agents.end(), std::back_inserter(agents_focus_region),
                 [](ABM<PotentialGermany<InfectionState>>::Agent a) {
                     return a.land == 8;
                 });
    agents.erase(std::remove_if(agents.begin(), agents.end(),
                                [](ABM<PotentialGermany<InfectionState>>::Agent a) {
                                    return a.land == 8;
                                }),
                 agents.end());

    //set adoption rates
    //set adoption rates for every federal state
    std::vector<AdoptionRate<Status>> adoption_rates;
    for (int i = 0; i < 16; i++) {
        adoption_rates.push_back({Status::S, Status::E, Region(i), 0.08, {Status::C, Status::I}, {1, 1}});
        adoption_rates.push_back({Status::E, Status::C, Region(i), 0.33});
        adoption_rates.push_back({Status::C, Status::I, Region(i), 0.36});
        adoption_rates.push_back({Status::C, Status::R, Region(i), 0.09});
        adoption_rates.push_back({Status::I, Status::D, Region(i), 0.001});
        adoption_rates.push_back({Status::I, Status::R, Region(i), 0.12});
    }

    ABM<PotentialGermany<InfectionState>> abm(agents_focus_region, adoption_rates, potential, metaregions);

    const unsigned regions = 16;

    SMModel<regions, Status> smm;

    //TODO: estimate transition rates due to abm sim
    std::vector<TransitionRate<Status>> transition_rates;
    ScalarType kappa = 0.01;
    for (int i = 0; i < 16; ++i) {
        for (int j = 0; j < 16; ++j) {
            if (i != j) {
                transition_rates.push_back({Status::S, Region(i), Region(j), 0.1 * kappa});
                transition_rates.push_back({Status::E, Region(i), Region(j), 0.1 * kappa});
                transition_rates.push_back({Status::C, Region(i), Region(j), 0.1 * kappa});
                transition_rates.push_back({Status::I, Region(i), Region(j), 0.1 * kappa});
                transition_rates.push_back({Status::R, Region(i), Region(j), 0.1 * kappa});
            }
        }
    }

    smm.parameters.get<AdoptionRates<Status>>()   = adoption_rates;
    smm.parameters.get<TransitionRates<Status>>() = transition_rates;

    //set populations for smm
    std::vector<std::vector<ScalarType>> populations;
    for (int i = 0; i < regions; ++i) {
        std::vector<ScalarType> pop(static_cast<size_t>(Status::Count));
        if (i != 8) {
            for (size_t s = 0; s < pop.size(); ++s) {
                pop[s] =
                    std::count_if(agents.begin(), agents.end(), [i, s](ABM<PotentialGermany<InfectionState>>::Agent a) {
                        return (a.land == i && a.status == Status(s));
                    });
            }
        }
        populations.push_back(pop);
    }

    for (size_t k = 0; k < regions; k++) {
        for (int i = 0; i < static_cast<size_t>(Status::Count); i++) {
            smm.populations[{static_cast<Region>(k), static_cast<Status>(i)}] = populations[k][i];
        }
    }

    PDMModel<regions, Status> pdmm;
    pdmm.parameters.get<AdoptionRates<Status>>()   = smm.parameters.get<AdoptionRates<Status>>();
    pdmm.parameters.get<TransitionRates<Status>>() = smm.parameters.get<TransitionRates<Status>>();
    pdmm.populations                               = smm.populations;

    double delta_exchange_time = 0.2;
    double start_time          = 0.0;
    double end_time            = 5.0;

    auto simABM  = mio::Simulation<ABM<PotentialGermany<InfectionState>>>(abm, start_time, 0.05);
    auto simPDMM = mio::Simulation<PDMModel<regions, Status>>(pdmm, start_time, 0.05);

    TIME_TYPE pre = TIME_NOW;

    for (double t = start_time; t < end_time; t = std::min(t + delta_exchange_time, end_time)) {
        printf("%.1f/%.1f\r", t, end_time);
        simABM.advance(t);
        simPDMM.advance(t);
        { //move agents from abm to pdmm
            auto& agents = simABM.get_model().populations;
            auto itr     = agents.begin();
            while (itr != agents.end()) {
                if (itr->land != 8) {
                    simPDMM.get_model().populations[{mio::mpm::Region(itr->land), itr->status}] += 1;
                    itr = agents.erase(itr);
                }
                else {
                    itr++;
                }
            }
        }
        { //move agents from abm to pdmm
            auto& pop = simPDMM.get_model().populations;
            for (int i = 0; i < (int)Status::Count; i++) {
                for (auto& agents = pop[{mio::mpm::Region(8), (Status)i}]; agents > 0; agents -= 1) {
                    //TODO: put agent to center of focus region
                    simABM.get_model().populations.push_back({{420, 765}, (Status)i, 8});
                }
            }
        }
    }

    TIME_TYPE post = TIME_NOW;

    fprintf(stdout, "# Elapsed time during advance(): %.*g\n", PRECISION, PRINTABLE_TIME(post - pre));

    std::vector<std::string> comps(16 * int(Status::Count));
    for (int i = 0; i < 16; ++i) {
        std::vector<std::string> c = {"S", "E", "C", "I", "R", "D"};
        std::copy(c.begin(), c.end(), comps.begin() + i * int(Status::Count));
    }

    FILE* outfile1 = fopen("~/results/outputABM.txt", "w");
    mio::mpm::print_to_file(outfile1, simABM.get_result(), comps);
    fclose(outfile1);
    FILE* outfile2 = fopen("~/results/outputPDMM.txt", "w");
    mio::mpm::print_to_file(outfile2, simPDMM.get_result(), comps);
    fclose(outfile2);

    return 0;
}