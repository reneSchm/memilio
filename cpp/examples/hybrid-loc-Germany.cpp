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
#include "memilio/data/analyze_result.h"

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
void read_initialization(std::string filename, std::vector<Agent>& agents)
{
    auto result = mio::read_json(filename).value();
    for (int i = 0; i < result.size(); ++i) {
        auto a = mio::deserialize_json(result[std::to_string(i)], mio::Tag<Agent>{}).value();
        agents.push_back(Agent{a.position, a.status, a.land});
    }
}

void run_simulation(std::string init_file, std::vector<mio::mpm::AdoptionRate<InfectionState>>& adoption_rates,
                    std::vector<mio::mpm::TransitionRate<InfectionState>>& transition_rates, Eigen::MatrixXd& potential,
                    Eigen::MatrixXi& metaregions, double tmax, double delta_t)
{
    const unsigned regions = 16;
    int focus_region       = 8;

    //read agents
    std::vector<mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent> agents;
    std::cerr << "begin read\n" << std::flush;
    TIME_TYPE pre_read = TIME_NOW;
    read_initialization<mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent>(init_file, agents);

    TIME_TYPE post_read = TIME_NOW;
    fprintf(stdout, "# Time for read init: %.*g\n", PRECISION, PRINTABLE_TIME(post_read - pre_read));
    std::cerr << "end read\n" << std::flush;

    int num_agents = agents.size();

    std::vector<mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent> agents_focus_region;
    std::copy_if(agents.begin(), agents.end(), std::back_inserter(agents_focus_region),
                 [focus_region](mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent a) {
                     return a.land == focus_region;
                 });
    agents.erase(std::remove_if(agents.begin(), agents.end(),
                                [focus_region](mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent a) {
                                    return a.land == focus_region;
                                }),
                 agents.end());
    mio::mpm::ABM<PotentialGermany<InfectionState>> abm(agents_focus_region, adoption_rates, potential, metaregions);

    std::vector<std::vector<ScalarType>> populations;
    for (int i = 0; i < regions; ++i) {
        std::vector<ScalarType> pop(static_cast<size_t>(InfectionState::Count));
        if (i != 8) {
            for (size_t s = 0; s < pop.size(); ++s) {
                pop[s] = std::count_if(agents.begin(), agents.end(),
                                       [i, s](mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent a) {
                                           return (a.land == i && a.status == InfectionState(s));
                                       });
            }
        }
        populations.push_back(pop);
    }

    //delete adoption rates for focus reagion i.e. set rates for focus region to 0
    adoption_rates.erase(std::remove_if(adoption_rates.begin(), adoption_rates.end(),
                                        [focus_region](mio::mpm::AdoptionRate<InfectionState> rate) {
                                            return rate.region == mio::mpm::Region(focus_region);
                                        }));
    mio::mpm::SMModel<regions, InfectionState> smm;
    smm.parameters.get<mio::mpm::AdoptionRates<InfectionState>>()   = adoption_rates;
    smm.parameters.get<mio::mpm::TransitionRates<InfectionState>>() = transition_rates;
    for (size_t k = 0; k < regions; k++) {
        for (int i = 0; i < static_cast<size_t>(InfectionState::Count); i++) {
            smm.populations[{static_cast<mio::mpm::Region>(k), static_cast<InfectionState>(i)}] = populations[k][i];
        }
    }
    mio::mpm::PDMModel<regions, InfectionState> pdmm;
    pdmm.parameters.get<mio::mpm::AdoptionRates<InfectionState>>() =
        smm.parameters.get<mio::mpm::AdoptionRates<InfectionState>>();
    pdmm.parameters.get<mio::mpm::TransitionRates<InfectionState>>() =
        smm.parameters.get<mio::mpm::TransitionRates<InfectionState>>();
    pdmm.populations = smm.populations;

    auto simABM  = mio::Simulation<mio::mpm::ABM<PotentialGermany<InfectionState>>>(abm, 0.0, delta_t);
    auto simPDMM = mio::Simulation<mio::mpm::PDMModel<regions, InfectionState>>(pdmm, 0.0, delta_t);

    double delta_exchange_time = 0.1;

    std::cerr << "begin simulate\n" << std::flush;

    TIME_TYPE pre = TIME_NOW;

    for (double t = 0.; t < tmax; t = std::min(t + delta_exchange_time, tmax)) {
        simABM.advance(t);
        simPDMM.advance(t);
        { //move agents from abm to pdmm
            auto& agents = simABM.get_model().populations;
            auto itr     = agents.begin();
            while (itr != agents.end()) {
                if (itr->land != 8) {
                    //simPDMM.get_result().get_last_value()[m_model->populations.get_flat_index({rate.from, rate.status})] -= 1;
                    simPDMM.get_result().get_last_value()[simPDMM.get_model().populations.get_flat_index(
                        {mio::mpm::Region(itr->land), itr->status})] += 1;
                    simPDMM.get_model().populations[{mio::mpm::Region(itr->land), itr->status}] += 1;
                    itr = agents.erase(itr);
                }
                else {
                    itr++;
                }
            }
        }
        { //move agents from abm to pdmm
            auto& pop = simPDMM.get_result().get_last_value();
            for (int i = 0; i < (int)InfectionState::Count; i++) {
                auto& p = pop[simPDMM.get_model().populations.get_flat_index({mio::mpm::Region(8), (InfectionState)i})];
                if (p > 0) {
                    if (floor(p) != p) {
                        std::cout << "p is not whole\n";
                    }
                }
                for (auto agents =
                         pop[simPDMM.get_model().populations.get_flat_index({mio::mpm::Region(8), (InfectionState)i})];
                     agents > 0; --agents) {
                    // auto& val = simPDMM.get_result().get_last_value()[simPDMM.get_model().populations.get_flat_index(
                    //     {mio::mpm::Region(8), (InfectionState)i})];
                    simABM.get_model().populations.push_back({{420, 765}, (InfectionState)i, focus_region});
                    simPDMM.get_result().get_last_value()[simPDMM.get_model().populations.get_flat_index(
                        {mio::mpm::Region(8), (InfectionState)i})] -= 1;
                }
            }
        }
    }

    TIME_TYPE post = TIME_NOW;

    std::cerr << "end simulate\n" << std::flush;

    fprintf(stdout, "# Hybrid model: Time for %.14f agents: %.*g\n", float(num_agents), PRECISION,
            PRINTABLE_TIME(post - pre));

    std::vector<std::string> comps(16 * int(InfectionState::Count));
    for (int i = 0; i < 16; ++i) {
        std::vector<std::string> c = {"S", "E", "C", "I", "R", "D"};
        std::copy(c.begin(), c.end(), comps.begin() + i * int(InfectionState::Count));
    }

    auto outpath_abm_int  = "/outputABMHybrid" + std::to_string(num_agents) + "_interpolated.txt";
    auto outpath_pdmm_int = "/outputPDMMHybrid" + std::to_string(num_agents) + "_interpolated.txt";
    auto outpath_abm      = "/outputABMHybrid" + std::to_string(num_agents) + ".txt";
    auto outpath_pdmm     = "/outputPDMMHybrid" + std::to_string(num_agents) + ".txt";
    FILE* outfile1        = fopen(outpath_abm.c_str(), "w");
    mio::mpm::print_to_file(outfile1, simABM.get_result(), comps);
    fclose(outfile1);
    FILE* outfile2 = fopen(outpath_pdmm.c_str(), "w");
    mio::mpm::print_to_file(outfile2, simPDMM.get_result(), comps);
    fclose(outfile2);
    FILE* outfile3 = fopen(outpath_abm_int.c_str(), "w");
    mio::mpm::print_to_file(outfile3, mio::interpolate_simulation_result(simABM.get_result()), comps);
    fclose(outfile3);
    FILE* outfile4 = fopen(outpath_pdmm_int.c_str(), "w");
    mio::mpm::print_to_file(outfile4, mio::interpolate_simulation_result(simPDMM.get_result()), comps);
    fclose(outfile4);
}

int main(int argc, char** argv)
{
    mio::set_log_level(mio::LogLevel::warn);
    using namespace mio::mpm;
    using Status = ABM<PotentialGermany<InfectionState>>::Status;

    Eigen::MatrixXd potential;
    Eigen::MatrixXi metaregions;

    std::cerr << "Setup: Read potential.\n" << std::flush;
    {
        const auto fname = "/potentially_germany.pgm";
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
        const auto fname = "/metagermany.pgm";
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

    //set adoption rates
    //set adoption rates for every federal state
    std::vector<AdoptionRate<Status>> adoption_rates;
    for (int i = 0; i < 16; i++) {
        adoption_rates.push_back({Status::S, Status::E, Region(i), 0.299352, {Status::C, Status::I}, {1, 1}});
        adoption_rates.push_back({Status::E, Status::C, Region(i), 0.33});
        adoption_rates.push_back({Status::C, Status::I, Region(i), 0.36});
        adoption_rates.push_back({Status::C, Status::R, Region(i), 0.09});
        adoption_rates.push_back({Status::I, Status::D, Region(i), 0.001});
        adoption_rates.push_back({Status::I, Status::R, Region(i), 0.12});
    }

    std::vector<TransitionRate<Status>> transition_rates;
    std::vector<Status> transitioning_states{Status::S, Status::E, Status::C, Status::I, Status::R};
    std::map<std::tuple<Region, Region>, double> factors{
        {{Region(0), Region(1)}, 0.0116313},     {{Region(0), Region(2)}, 0.0494058},
        {{Region(0), Region(4)}, 0.0262153},     {{Region(1), Region(0)}, 0.00938663},
        {{Region(1), Region(4)}, 0.00487336},    {{Region(1), Region(5)}, 0.0328892},
        {{Region(2), Region(0)}, 0.0621894},     {{Region(2), Region(4)}, 0.0516505},
        {{Region(3), Region(4)}, 0.0437042},     {{Region(4), Region(0)}, 0.0254591},
        {{Region(4), Region(1)}, 0.00610971},    {{Region(4), Region(2)}, 0.0448326},
        {{Region(4), Region(3)}, 0.0361901},     {{Region(4), Region(5)}, 0.000624175},
        {{Region(4), Region(7)}, 0.0131797},     {{Region(4), Region(8)}, 0.0132997},
        {{Region(4), Region(10)}, 0.0021486},    {{Region(4), Region(11)}, 0.000984276},
        {{Region(5), Region(1)}, 0.0359261},     {{Region(5), Region(4)}, 0.000480134},
        {{Region(5), Region(6)}, 0.153343},      {{Region(5), Region(7)}, 0.0222662},
        {{Region(5), Region(9)}, 0.0154723},     {{Region(6), Region(5)}, 0.194719},
        {{Region(7), Region(4)}, 0.0131677},     {{Region(7), Region(5)}, 0.0206938},
        {{Region(7), Region(9)}, 0.0127716},     {{Region(7), Region(10)}, 0.0218821},
        {{Region(8), Region(4)}, 0.0},           {{Region(8), Region(11)}, 0.0},
        {{Region(8), Region(12)}, 0.0},          {{Region(9), Region(5)}, 0.0},
        {{Region(9), Region(7)}, 0.0126395},     {{Region(9), Region(10)}, 0.0157604},
        {{Region(9), Region(14)}, 0.000504141},  {{Region(10), Region(4)}, 0.00289281},
        {{Region(10), Region(7)}, 0.0223142},    {{Region(10), Region(9)}, 0.014332},
        {{Region(10), Region(11)}, 0.010575},    {{Region(10), Region(14)}, 0.0127956},
        {{Region(11), Region(4)}, 0.0000984276}, {{Region(11), Region(8)}, 0.0102029},
        {{Region(11), Region(10)}, 0.00972272},  {{Region(11), Region(12)}, 0.0475093},
        {{Region(11), Region(14)}, 0.0483255},   {{Region(11), Region(15)}, 0.0064458},
        {{Region(12), Region(8)}, 0.0122674},    {{Region(12), Region(11)}, 0.0433201},
        {{Region(12), Region(13)}, 0.0362502},   {{Region(12), Region(15)}, 0.00447725},
        {{Region(13), Region(12)}, 0.0403793},   {{Region(14), Region(9)}, 0.000372104},
        {{Region(14), Region(10)}, 0.010851},    {{Region(14), Region(11)}, 0.0403553},
        {{Region(14), Region(15)}, 0.0218821},   {{Region(15), Region(11)}, 0.00590565},
        {{Region(15), Region(12)}, 0.00650582},  {{Region(15), Region(14)}, 0.0246309}};
    ScalarType kappa = 0.01;
    for (auto& rate : factors) {
        for (auto state : transitioning_states) {
            transition_rates.push_back({state, std::get<0>(rate.first), std::get<1>(rate.first), rate.second * kappa});
        }
    }

    run_simulation("/initialization10000.json", adoption_rates, transition_rates, potential, metaregions, 100.0, 0.05);

    return 0;
}