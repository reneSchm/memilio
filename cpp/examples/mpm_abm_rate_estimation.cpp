#include "hybrid_paper/lib/potentials/potential_germany.h"
#include "hybrid_paper/lib/map_reader.h"
#include "mpm/abm.h"
#include "mpm/utility.h"

#include "memilio/data/analyze_result.h"
#include "memilio/io/json_serializer.h"

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

template <class Status = InfectionState>
std::string to_string(Status state)
{
    std::string string_state;
    switch (state) {
    case InfectionState(0):
        string_state = "S";
        break;

    case InfectionState(1):
        string_state = "E";
        break;

    case InfectionState(2):
        string_state = "C";
        break;

    case InfectionState(3):
        string_state = "I";
        break;

    case InfectionState(4):
        string_state = "R";
        break;

    case InfectionState(5):
        string_state = "D";
        break;

    default:
        string_state = "state not found";
        break;
    }
    return string_state;
}

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
        a.region   = metaregions(pos_candidate[0], pos_candidate[1]) - 1;
        //only infected agents in focus region
        if (a.region == 8) {
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
    return mio::success();
}

template <class Agent, class Position>
mio::IOResult<void> create_initialization_for_Germany(std::vector<std::vector<double>>& pop_dists, std::string filename,
                                                      ScalarType scaling_factor, Eigen::MatrixXd potential,
                                                      Eigen::MatrixXi metaregions)
{
    std::vector<Agent> agents;
    auto data = mio::read_json(filename).value();
    std::vector<std::string> state_mapping{"Schleswig-Holstein",
                                           "Mecklenburg-Vorpommern",
                                           "Hamburg",
                                           "Bremen",
                                           "Niedersachsen",
                                           "Brandenburg",
                                           "Berlin",
                                           "Sachsen-Anhalt",
                                           "Nordrhein-Westfalen",
                                           "Sachsen",
                                           "Thueringen",
                                           "Hessen",
                                           "Rheinland-Pfalz",
                                           "Saarland",
                                           "Bayern",
                                           "Baden-Wuerttemberg"};
    std::vector<Position> centers{{645, 1203}, {912, 1128}, {669, 1095}, {561, 1023}, {645, 1005}, {996, 1002},
                                  {999, 945},  {837, 825},  {420, 765},  {1005, 708}, {771, 684},  {561, 591},
                                  {411, 525},  {366, 441},  {879, 368},  {576, 287}};
    for (size_t state = 0; state < pop_dists.size(); ++state) {
        double population = 0;
        //find population
        for (int d = 0; d < data.size(); ++d) {
            if (data[d]["State"].asString() == state_mapping[state]) {
                population = data[d]["Population"].asDouble() / scaling_factor;
                break;
            }
        }
        for (size_t agent = 0; agent < population; ++agent) {
            auto infection_state =
                static_cast<InfectionState>(mio::DiscreteDistribution<int>::get_instance()(pop_dists[state]));
            agents.push_back({centers[state], infection_state, int(state)});
        }
    }
    std::vector<mio::mpm::AdoptionRate<InfectionState>> adoption_rates{};

    // for (auto& agent : agents) {
    //     std::cout << agent.position[0] << " " << agent.position[1] << " ";
    // }
    // std::cout << "\n";

    mio::mpm::ABM<PotentialGermany<InfectionState>> model(agents, adoption_rates, potential, metaregions);
    auto sim = mio::Simulation<mio::mpm::ABM<PotentialGermany<InfectionState>>>(model, 0.0, 0.05);

    for (auto& agent : sim.get_model().populations) {
        if (agent.region == 2 || agent.region == 3 || agent.region == 6) {
            sim.get_model().move(0.0, 0.1, agent);
        }
        else {
            sim.get_model().move(0.0, 1.0, agent);
        }
    }

    Json::Value all_agents;
    for (int i = 0; i < sim.get_model().populations.size(); ++i) {
        BOOST_OUTCOME_TRY(a, mio::serialize_json(sim.get_model().populations[i]));
        all_agents[std::to_string(i)] = a;
    }
    // for (auto& agent : sim.get_model().populations) {
    //     std::cout << agent.position[0] << " " << agent.position[1] << " ";
    // }
    // std::cout << "\n";
    auto write_status = mio::write_json("initialization" + std::to_string(int(scaling_factor)) + ".json", all_agents);
    return mio::success();
}

template <class Agent>
void read_initialization(std::string filename, std::vector<Agent>& agents)
{
    auto result = mio::read_json(filename).value();
    for (int i = 0; i < result.size(); ++i) {
        auto a = mio::deserialize_json(result[std::to_string(i)], mio::Tag<Agent>{}).value();
        agents.push_back(Agent{a.position, a.status, a.region});
    }
}

template <class Status>
std::vector<mio::mpm::TransitionRate<Status>> add_transition_rates(std::vector<mio::mpm::TransitionRate<Status>>& v1,
                                                                   std::vector<mio::mpm::TransitionRate<Status>>& v2)
{
    std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(),
                   [](mio::mpm::TransitionRate<Status>& t1, mio::mpm::TransitionRate<Status>& t2) {
                       return mio::mpm::TransitionRate<Status>{t1.status, t1.from, t1.to, t1.factor + t2.factor};
                   });
    return v1;
}

template <class Status>
std::vector<mio::mpm::AdoptionRate<Status>> add_adoption_rates(std::vector<mio::mpm::AdoptionRate<Status>>& v1,
                                                               std::vector<mio::mpm::AdoptionRate<Status>>& v2)
{
    std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(),
                   [](mio::mpm::AdoptionRate<Status>& a1, mio::mpm::AdoptionRate<Status>& a2) {
                       return mio::mpm::AdoptionRate<Status>{a1.from,       a1.to,     a1.region, a1.factor + a2.factor,
                                                             a1.influences, a1.factors};
                   });
    return v1;
}

template <class Status>
void print_adoption_rates(std::vector<mio::mpm::AdoptionRate<Status>>& adoption_rates)
{
    std::cout << "\n Adoption Rates: \n";
    std::cout << "from to region factor \n";
    for (auto& rate : adoption_rates) {
        std::cout << to_string(rate.from) << "  " << to_string(rate.to) << "  " << rate.region << "  " << rate.factor
                  << "\n";
    }
}

template <class Status>
void print_transition_rates(std::vector<mio::mpm::TransitionRate<Status>>& transition_rates)
{
    std::cout << "\n Transition Rates: \n";
    std::cout << "status from to factor \n";
    for (auto& rate : transition_rates) {
        std::cout << to_string(rate.status) << "  " << rate.from << "  " << rate.to << "  " << rate.factor << "\n";
    }
}

void calculate_rates_for_mpm(mio::mpm::ABM<PotentialGermany<InfectionState>>& abm,
                             std::vector<mio::mpm::AdoptionRate<InfectionState>>& adoption_rates_model, size_t n_runs,
                             double tmax)
{
    std::vector<std::vector<mio::mpm::AdoptionRate<InfectionState>>> estimated_adoption_rates(n_runs);
    std::vector<std::vector<mio::mpm::TransitionRate<InfectionState>>> estimated_transition_rates(n_runs);

    std::vector<mio::mpm::AdoptionRate<InfectionState>> zero_adoption_rates;
    std::vector<mio::mpm::TransitionRate<InfectionState>> zero_transition_rates;

    for (int i = 0; i < 16; ++i) {
        zero_adoption_rates.push_back({InfectionState::S,
                                       InfectionState::E,
                                       mio::mpm::Region(i),
                                       0.0,
                                       {InfectionState::C, InfectionState::I},
                                       {1, 1}});
        zero_adoption_rates.push_back({InfectionState::E, InfectionState::C, mio::mpm::Region(i), 0.0});
        zero_adoption_rates.push_back({InfectionState::C, InfectionState::I, mio::mpm::Region(i), 0.0});
        zero_adoption_rates.push_back({InfectionState::C, InfectionState::R, mio::mpm::Region(i), 0.0});
        zero_adoption_rates.push_back({InfectionState::I, InfectionState::D, mio::mpm::Region(i), 0.0});
        zero_adoption_rates.push_back({InfectionState::I, InfectionState::R, mio::mpm::Region(i), 0.0});
        for (int j = 0; j < 16; ++j) {
            if (i != j) {
                zero_transition_rates.push_back({InfectionState::S, mio::mpm::Region(i), mio::mpm::Region(j), 0});
                zero_transition_rates.push_back({InfectionState::E, mio::mpm::Region(i), mio::mpm::Region(j), 0});
                zero_transition_rates.push_back({InfectionState::C, mio::mpm::Region(i), mio::mpm::Region(j), 0});
                zero_transition_rates.push_back({InfectionState::I, mio::mpm::Region(i), mio::mpm::Region(j), 0});
                zero_transition_rates.push_back({InfectionState::R, mio::mpm::Region(i), mio::mpm::Region(j), 0});
            }
        }
    }
    std::fill_n(estimated_transition_rates.begin(), n_runs, zero_transition_rates);
    std::fill_n(estimated_adoption_rates.begin(), n_runs, zero_adoption_rates);

    for (size_t run = 0; run < n_runs; run++) {
        std::cerr << "run number: " << run << "\n" << std::flush;
        mio::Simulation<mio::mpm::ABM<PotentialGermany<InfectionState>>> sim(abm, 0, 0.05);
        sim.advance(tmax);

        //add calculated transition rates
        for (auto& adop_rate : estimated_adoption_rates[run]) {
            if (adop_rate.influences.size() == 0) {
                //first-order rate
                adop_rate.factor = (std::find_if(adoption_rates_model.begin(), adoption_rates_model.end(),
                                                 [adop_rate](mio::mpm::AdoptionRate<InfectionState> rate) {
                                                     return (rate.from == adop_rate.from && rate.to == adop_rate.to);
                                                 })
                                        ->factor);
            }
            else {
                //second-order rate
                adop_rate.factor = sim.get_model().accumulated_contact_rates / sim.get_model().contact_rates_count *
                                   (std::find_if(adoption_rates_model.begin(), adoption_rates_model.end(),
                                                 [adop_rate](mio::mpm::AdoptionRate<InfectionState> rate) {
                                                     return (rate.from == adop_rate.from && rate.to == adop_rate.to);
                                                 })
                                        ->factor);
            }
        }

        //add calculated transition rates
        for (auto& tr_rate : estimated_transition_rates[run]) {
            tr_rate.factor = sim.get_model().number_transitions(tr_rate) / (abm.populations.size() * tmax);
        }
    }

    auto mean_transition_rates = std::accumulate(estimated_transition_rates.begin(), estimated_transition_rates.end(),
                                                 zero_transition_rates, add_transition_rates<InfectionState>);

    auto mean_adoption_rates = std::accumulate(estimated_adoption_rates.begin(), estimated_adoption_rates.end(),
                                               zero_adoption_rates, add_adoption_rates<InfectionState>);

    double denominator{(1 / (double)n_runs)};
    std::transform(
        mean_transition_rates.begin(), mean_transition_rates.end(), mean_transition_rates.begin(),
        [&denominator](mio::mpm::TransitionRate<InfectionState>& rate) {
            return mio::mpm::TransitionRate<InfectionState>{rate.status, rate.from, rate.to, denominator * rate.factor};
        });
    std::transform(mean_adoption_rates.begin(), mean_adoption_rates.end(), mean_adoption_rates.begin(),
                   [&denominator](mio::mpm::AdoptionRate<InfectionState>& rate) {
                       return mio::mpm::AdoptionRate<InfectionState>{
                           rate.from, rate.to, rate.region, denominator * rate.factor, rate.influences, rate.factors};
                   });

    print_adoption_rates(mean_adoption_rates);
    print_transition_rates(mean_transition_rates);
}

void get_agent_movement(size_t n_agents, Eigen::MatrixXd potential, Eigen::MatrixXi metaregions)
{
    using Position = mio::mpm::ABM<PotentialGermany<InfectionState>>::Position;
    std::vector<mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent> agents(n_agents);

    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    auto& sta_rng = mio::DiscreteDistribution<int>::get_instance();
    for (auto& a : agents) {
        Eigen::Vector2d pos_candidate{pos_rng(0.0, double(potential.rows())), pos_rng(0.0, double(potential.cols()))};
        while (metaregions(pos_candidate[0], pos_candidate[1]) == 0 ||
               potential(pos_candidate[0], pos_candidate[1]) != 0) {
            pos_candidate = {pos_rng(0.0, double(potential.rows())), pos_rng(0.0, double(potential.cols()))};
        }
        a.position = {pos_candidate[0], pos_candidate[1]};
        a.region   = metaregions(pos_candidate[0], pos_candidate[1]) - 1;
        a.status   = InfectionState::S;
    }
    std::vector<mio::mpm::AdoptionRate<InfectionState>> adoption_rates{};

    mio::mpm::ABM<PotentialGermany<InfectionState>> model(agents, adoption_rates, potential, metaregions);
    const double dt   = 0.05;
    double t          = 0;
    const double tmax = 100;

    auto sim = mio::Simulation<mio::mpm::ABM<PotentialGermany<InfectionState>>>(model, t, 0.05);
    std::vector<std::vector<Position>> result;
    //get start positions
    std::vector<Position> positions;
    for (auto& agent : sim.get_model().populations) {
        positions.push_back(agent.position);
    }
    result.push_back(positions);
    while (t < tmax) {
        positions.clear();
        for (auto& agent : sim.get_model().populations) {
            sim.get_model().move(t, dt, agent);
            positions.push_back(agent.position);
        }
        result.push_back(positions);
        t += dt;
    }

    //write result to file
    FILE* outfile = fopen("~/results/positions.txt", "w");
    for (size_t time = 0; time < result.size(); ++time) {
        fprintf(outfile, "\n%s", "t");
        auto res_t = result[time];
        for (size_t a = 0; a < res_t.size(); ++a) {
            std::string p = "(" + std::to_string(res_t[a][0]) + "," + std::to_string(res_t[a][1]) + ")";
            fprintf(outfile, " %s", p.c_str());
        }
        //fprintf(outfile, "\n%s");
    }
    fclose(outfile);
}

void run_simulation(std::string init_file, std::vector<mio::mpm::AdoptionRate<InfectionState>>& adoption_rates,
                    Eigen::MatrixXd& potential, Eigen::MatrixXi& metaregions, double tmax, double delta_t)
{
    std::vector<mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent> agents;
    TIME_TYPE pre_read = TIME_NOW;
    read_initialization<mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent>(init_file, agents);

    TIME_TYPE post_read = TIME_NOW;
    fprintf(stdout, "# Time for read init: %.*g\n", PRECISION, PRINTABLE_TIME(post_read - pre_read));

    int num_agents = agents.size();
    //create model
    mio::mpm::ABM<PotentialGermany<InfectionState>> model(agents, adoption_rates, potential, metaregions);

    TIME_TYPE pre_sim = TIME_NOW;
    auto result       = mio::simulate(0, tmax, delta_t, model);

    TIME_TYPE post_sim = TIME_NOW;

    fprintf(stdout, "# Time for %.14f agents: %.*g\n", float(num_agents), PRECISION,
            PRINTABLE_TIME(post_sim - pre_sim));

    std::vector<std::string> comps(16 * int(InfectionState::Count));
    for (int i = 0; i < 16; ++i) {
        std::vector<std::string> c = {"S", "E", "C", "I", "R", "D"};
        std::copy(c.begin(), c.end(), comps.begin() + i * int(InfectionState::Count));
    }

    auto outpath  = "~/results/outputABMSim" + std::to_string(num_agents) + ".txt";
    FILE* outfile = fopen(outpath.c_str(), "w");
    mio::mpm::print_to_file(outfile, result, comps);
    fclose(outfile);
    auto outpath_int = "~/results/outputABMSim" + std::to_string(num_agents) + "_interpolated.txt";
    FILE* outfile1   = fopen(outpath_int.c_str(), "w");
    mio::mpm::print_to_file(outfile1, mio::interpolate_simulation_result(result), comps);
    fclose(outfile1);
}

int main()
{
    //mio::thread_local_rng().seed({114381446, 2427727386, 806223567, 832414962, 4121923627, 1581162203});
    using namespace mio::mpm;
    using Status    = ABM<PotentialGermany<InfectionState>>::Status;
    using Position  = mio::mpm::ABM<PotentialGermany<InfectionState>>::Position;
    size_t n_agents = 16 * 100;

    Eigen::MatrixXd potential;
    Eigen::MatrixXi metaregions;

    std::cerr << "Setup: Read potential.\n" << std::flush;
    {
        const auto fname =
            "C:/Users/bick_ju/Documents/repos/hybrid/example-hybrid/data/potential/potentially_germany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            return 1;
        }
        else {
            potential = 8 * mio::mpm::read_pgm(ifile);
            ifile.close();
        }
    }
    std::cerr << "Setup: Read metaregions.\n" << std::flush;
    {
        const auto fname = "C:/Users/bick_ju/Documents/repos/hybrid/example-hybrid/data/potential/metagermany.pgm";
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

    // std::vector<std::vector<double>> pop_dists{
    //     {1.0, 0.0, 0.0, 0.0, 0.0},    {1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0},
    //     {1.0, 0.0, 0.0, 0.0, 0.0},    {1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0},
    //     {0.9, 0.04, 0.05, 0.01, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0},
    //     {1.0, 0.0, 0.0, 0.0, 0.0},    {1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0}, {1.0, 0.0, 0.0, 0.0, 0.0}};

    // create_initialization_for_Germany<ABM<PotentialGermany<InfectionState>>::Agent, Position>(
    //     pop_dists, "C:/Users/bick_ju/Documents/results/population_data_states.json", 8000, potential, metaregions);

    // std::cerr << "Finished\n" << std::flush;
    //get_agent_movement(10, potential, metaregions);

    //std::vector<ABM<PotentialGermany<InfectionState>>::Agent> agents;

    // //std::vector<double> pop_dist{0.9, 0.05, 0.05, 0.0, 0.0};
    // //create_start_initialization<Status, ABM<PotentialGermany>::Agent>(agents, pop_dist, potential, metaregions);
    //set adoption rates for every federal state
    std::vector<AdoptionRate<Status>> adoption_rates;
    for (int i = 0; i < 16; i++) {
        adoption_rates.push_back({Status::S, Status::E, Region(i), 0.3, {Status::C, Status::I}, {1, 1}});
        adoption_rates.push_back({Status::E, Status::C, Region(i), 0.33});
        adoption_rates.push_back({Status::C, Status::I, Region(i), 0.36});
        adoption_rates.push_back({Status::C, Status::R, Region(i), 0.09});
        adoption_rates.push_back({Status::I, Status::D, Region(i), 0.001});
        adoption_rates.push_back({Status::I, Status::R, Region(i), 0.12});
    }

    run_simulation("~/input/initialization10000.json", adoption_rates, potential, metaregions, 100.0, 0.05);
    //std::vector<mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent> agents;
    //read_initialization<mio::mpm::ABM<PotentialGermany<InfectionState>>::Agent>(
    //    "C:/Users/bick_ju/Documents/results/agent_init/initialization10000.json", agents);

    //create model
    //mio::mpm::ABM<PotentialGermany<InfectionState>> model(agents, adoption_rates, potential, metaregions);

    // ABM<PotentialGermany<InfectionState>> model(agents, adoption_rates, potential, metaregions);
    //calculate_rates_for_mpm(model, adoption_rates, 10, 100);
    // std::cerr << "Starting simulation.\n" << std::flush;

    // auto result = mio::simulate(0, 100, 0.05, model);
    // std::vector<std::string> comps(16 * int(Status::Count));
    // for (int i = 0; i < 16; ++i) {
    //     std::vector<std::string> c = {"S", "E", "C", "I", "R", "D"};
    //     std::copy(c.begin(), c.end(), comps.begin() + i * int(Status::Count));
    // }

    // FILE* outfile1 = fopen("~/results/outputABMSim.txt", "w");
    // mio::mpm::print_to_file(outfile1, result, comps);
    // fclose(outfile1);
    // //mio::mpm::print_to_terminal(result, comps);

    return 0;
}