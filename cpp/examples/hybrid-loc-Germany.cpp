#include "memilio/compartments/simulation.h"
#include "memilio/config.h"
#include "memilio/utils/logging.h"
#include "memilio/io/json_serializer.h"
#include "memilio/utils/time_series.h"
#include "mpm/abm.h"
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

class PotentialGermany
{
    constexpr static double contact_radius = 0.6;

public:
    using Status   = InfectionState;
    using Position = Eigen::Vector2d;

    struct Agent {
        Position position;
        InfectionState status;
        int land;

        /**
     * serialize agents. 
     * @see mio::serialize
     */
        template <class IOContext>
        void serialize(IOContext& io) const
        {
            auto obj = io.create_object("Agent");
            obj.add_element("Position", position);
            obj.add_element("Status", status);
            obj.add_element("Land", land);
        }

        /**
     * deserialize an object of this class.
     * @see mio::deserialize
     */
        template <class IOContext>
        static mio::IOResult<Agent> deserialize(IOContext& io)
        {
            auto obj   = io.expect_object("Agent");
            auto pos   = obj.expect_element("Position", mio::Tag<Position>{});
            auto state = obj.expect_element("Status", mio::Tag<InfectionState>{});
            auto land  = obj.expect_element("Land", mio::Tag<int>{});
            return apply(
                io,
                [](auto&& pos_, auto&& state_, auto&& land_) {
                    return Agent{pos_, state_, land_};
                },
                pos, state, land);
        }
    };

    PotentialGermany(const std::vector<Agent>& agents, const typename mio::mpm::AdoptionRates<Status>::Type& rates,
                     Eigen::Ref<const Eigen::MatrixXd> potential, Eigen::Ref<const Eigen::MatrixXi> metaregions)
        : potential(potential)
        , metaregions(metaregions)
        , populations(agents)
        , sigma((double)potential.cols() / 200)
    {
        for (auto& agent : populations) {
            assert(is_in_domain(agent.position));
        }
        for (auto& r : rates) {
            m_adoption_rates.emplace(std::forward_as_tuple(r.region, r.from, r.to), r);
            //m_adoption_rates[{r.region, r.from, r.to}] = r;
        }
    }

    inline static constexpr void adopt(Agent& agent, const Status& new_status)
    {
        agent.status = new_status;
    }

    double adoption_rate(const Agent& agent, const Status& new_status) const
    {
        double rate = 0;
        // get the correct adoption rate
        // auto well    = (agent.position[0] < 0) ? 0 : 1;
        auto map_itr = m_adoption_rates.find({agent.land, agent.status, new_status});
        if (map_itr != m_adoption_rates.end()) {
            const auto& adoption_rate = map_itr->second;
            // calculate the current rate, depending on order
            if (adoption_rate.influences.size() == 0) { // first order adoption
                // contact independant rate
                rate = adoption_rate.factor;
            }
            else { // second order adoption
                // accumulate rate per contact with a status in influences
                size_t num_contacts   = 0;
                ScalarType influences = 0;
                for (auto& contact : populations) {
                    // check if contact is indeed a contact
                    if (is_contact(agent, contact)) {
                        num_contacts++;
                        for (size_t i = 0; i < adoption_rate.influences.size(); i++) {
                            if (contact.status == adoption_rate.influences[i]) {
                                influences += adoption_rate.factors[i];
                            }
                        }
                    }
                }
                // rate = factor * "concentration of contacts with status new_status"
                if (num_contacts > 0) {
                    rate = adoption_rate.factor * (influences / num_contacts);
                }
            }
        }
        // else: no adoption from agent.status to new_status exist
        return rate;
    }

    void move(const double t, const double dt, Agent& agent)
    {
        Position p = {mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0),
                      mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0)};

        auto pold       = agent.position;
        agent.position  = agent.position - dt * grad_U(agent.position) + (sigma * std::sqrt(dt)) * p;
        const auto land = metaregions(agent.position[0], agent.position[1]);
        if (land > 0) {
            agent.land = land - 1;
        }
    }

    Eigen::VectorXd time_point() const
    {
        // for (auto agent : populations) {
        //     std::cout << agent.position[0] << " " << agent.position[1] << " ";
        // }
        // std::cout << "\n";
        Eigen::Matrix<double, 16 * static_cast<size_t>(Status::Count), 1> val;
        val.setZero();
        for (auto& agent : populations) {
            val[(agent.land * static_cast<size_t>(Status::Count) + static_cast<size_t>(agent.status))]++;
        }
        return val;
    }

    std::vector<Agent> populations;

private:
    // static Position grad_U(const Position x)
    // {
    //     // U is a double well potential
    //     // U(x0,x1) = (x0^2 - 1)^2 + 2*x1^2
    //     return {4 * x[0] * (x[0] * x[0] - 1), 4 * x[1]};
    // }

    // discrete germany (bilinear)
    // Position grad_U(const Position p)
    // {
    //     Position x_shift = {1, 0}, y_shift = {0, 1};
    //     const ScalarType dUdx =
    //         (interpolate_bilinear(potential, p + x_shift) - interpolate_bilinear(potential, p - x_shift)) / 2;
    //     const ScalarType dUdy =
    //         (interpolate_bilinear(potential, p + y_shift) - interpolate_bilinear(potential, p - y_shift)) / 2;
    //     return {dUdx, dUdy};
    // }

    // discrete germany (nearest neighbour)
    Position grad_U(const Position p)
    {
        size_t x = std::round(p[0]), y = std::round(p[1]); // take floor of each coordinate
        const ScalarType dUdx = (potential(x + 1, y) - potential(x - 1, y)) / 2 * sigma;
        const ScalarType dUdy = (potential(x, y + 1) - potential(x, y - 1)) / 2 * sigma;
        return {dUdx, dUdy};
    }

    static ScalarType interpolate_bilinear(Eigen::Ref<const Eigen::MatrixXd> data, const Position& p)
    {
        size_t x = p[0], y = p[1]; // take floor of each coordinate
        const ScalarType tx = p[0] - x;
        const ScalarType ty = p[1] - y;
        // data point distance is always one, since we use integer indices
        const ScalarType bot = tx * data(x, y) + (1 - tx) * data(x + 1, y);
        const ScalarType top = tx * data(x, y + 1) + (1 - tx) * data(x + 1, y + 1);
        return ty * bot + (1 - ty) * top;
    }

    bool is_contact(const Agent& agent, const Agent& contact) const
    {
        return (&agent != &contact) && // test if agent and contact are different objects
               (agent.land == contact.land) && // test if they are in the same metaregion
               (agent.position - contact.position).norm() < contact_radius; // test if contact is in the contact radius
    }

    bool is_in_domain(const Position& p) const
    {
        // restrict domain to [-2, 2]^2 where "escaping" is impossible, i.e. it holds x <= grad_U(x)
        return 0 <= p[0] && p[0] <= potential.rows() && 0 <= p[1] && p[1] <= potential.cols();
    }

    Eigen::Ref<const Eigen::MatrixXd> potential;
    Eigen::Ref<const Eigen::MatrixXi> metaregions;
    std::map<std::tuple<mio::mpm::Region, Status, Status>, mio::mpm::AdoptionRate<Status>> m_adoption_rates;
    const double sigma;
};

Eigen::MatrixXd read_pgm(std::istream& pgm_file)
{
    size_t height, width, color_range;
    std::string reader;
    std::stringstream parser;
    // ignore magic number (P2)
    std::getline(pgm_file, reader);
    // get dims
    std::getline(pgm_file, reader);
    parser.str(reader);
    parser >> width >> height;
    // get color range (max value for colors)
    std::getline(pgm_file, reader);
    parser.clear();
    parser.str(reader);
    parser >> color_range;
    // read image data
    // we assume (0,0) to be at the bottom left
    Eigen::MatrixXd data(width, height);
    for (size_t j = height; j > 0; j--) {
        std::getline(pgm_file, reader);
        parser.clear();
        parser.str(reader);
        for (size_t i = 0; i < width; i++) {
            parser >> data(i, j - 1);
            data(i, j - 1) = data(i, j - 1) / color_range;
        }
    }
    return data;
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
    using Status = ABM<PotentialGermany>::Status;

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
            potential = read_pgm(ifile);
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
            metaregions = (16 * read_pgm(ifile))
                              .unaryExpr([](double c) {
                                  return std::round(c);
                              })
                              .cast<int>();
            ifile.close();
        }
    }

    //read agents
    std::vector<ABM<PotentialGermany>::Agent> agents;
    read_initialization<ABM<PotentialGermany>::Agent>("initialization.json", agents, 16 * 100);

    std::vector<ABM<PotentialGermany>::Agent> agents_focus_region;
    std::copy_if(agents.begin(), agents.end(), std::back_inserter(agents_focus_region),
                 [](ABM<PotentialGermany>::Agent a) {
                     return a.land == 8;
                 });
    agents.erase(std::remove_if(agents.begin(), agents.end(),
                                [](ABM<PotentialGermany>::Agent a) {
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

    ABM<PotentialGermany> abm(agents_focus_region, adoption_rates, potential, metaregions);

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
                pop[s] = std::count_if(agents.begin(), agents.end(), [i, s](ABM<PotentialGermany>::Agent a) {
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

    auto simABM  = mio::Simulation<ABM<PotentialGermany>>(abm, start_time, 0.05);
    auto simPDMM = mio::Simulation<PDMModel<regions, Status>>(pdmm, start_time, 0.05);

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
                    simABM.get_model().populations.push_back({{370, 770}, (Status)i, 8});
                }
            }
        }
    }

    std::vector<std::string> comps(16 * int(Status::Count));
    for (int i = 0; i < 16; ++i) {
        std::vector<std::string> c = {"S", "E", "C", "I", "R", "D"};
        std::copy(c.begin(), c.end(), comps.begin() + i * int(Status::Count));
    }

    FILE* outfile1 = fopen("outputABM.txt", "w");
    mio::mpm::print_to_file(outfile1, simABM.get_result(), comps);
    fclose(outfile1);
    FILE* outfile2 = fopen("outputPDMM.txt", "w");
    mio::mpm::print_to_file(outfile2, simPDMM.get_result(), comps);
    fclose(outfile2);

    return 0;
}