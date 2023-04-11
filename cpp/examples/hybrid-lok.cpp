#include "memilio/compartments/simulation.h"
#include "memilio/config.h"
#include "memilio/utils/logging.h"
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
    I,
    R,
    Count
};

class QuadWellModel
{
    constexpr static double contact_radius = 0.4;
    constexpr static double sigma          = 0.5;

public:
    using Status   = InfectionState;
    using Position = Eigen::Vector2d;

    struct Agent {
        Position position;
        InfectionState status;
    };

    QuadWellModel(const std::list<Agent>& agents, const typename mio::mpm::AdoptionRates<Status>::Type& rates)
        : populations(agents)
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
        auto well    = (agent.position[0] < 0) ? 0 : 1;
        auto map_itr = m_adoption_rates.find({well, agent.status, new_status});
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

    static void move(const double t, const double dt, Agent& agent)
    {
        Position p = {mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0),
                      mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0)};

        agent.position = agent.position - dt * grad_U(agent.position) + (sigma * std::sqrt(dt)) * p;
    }

    Eigen::VectorXd time_point() const
    {
        Eigen::Matrix<double, 4 * static_cast<size_t>(Status::Count), 1> val;
        val.setZero();
        for (auto& agent : populations) {
            // split population into the wells given by grad_U
            auto position = (agent.position[0] < 0)
                                ? static_cast<size_t>(agent.status)
                                : static_cast<size_t>(agent.status) + static_cast<size_t>(Status::Count);
            position += (agent.position[1] > 0) ? 0 : 2 * static_cast<size_t>(Status::Count);
            val[position]++;
        }
        return val;
    }

    std::list<Agent> populations;

private:
    static Position grad_U(const Position x)
    {
        // U is a quad well potential
        // U(x0,x1) = (x0^2 - 1)^2 + (x1^2 - 1)^2
        return {4 * x[0] * (x[0] * x[0] - 1), 4 * x[1] * (x[1] * x[1] - 1)};
    }

    bool is_contact(const Agent& agent, const Agent& contact) const
    {
        //      test if contact is in the contact radius                   and test if agent and contact are different objects
        return (agent.position - contact.position).norm() < contact_radius && (&agent != &contact);
    }

    bool is_in_domain(const Position& p) const
    {
        // restrict domain to [-2, 2]^2 where "escaping" is impossible, i.e. it holds x <= grad_U(x) for dt <= 0.1
        return -2 <= p[0] && p[0] <= 2 && -2 <= p[1] && p[1] <= 2;
    }

    std::map<std::tuple<mio::mpm::Region, Status, Status>, mio::mpm::AdoptionRate<Status>> m_adoption_rates;
};

std::vector<ScalarType>& scale(std::vector<ScalarType>& v, ScalarType s)
{
    for (auto&& i : v)
        i *= s;
    return v;
}

int main(int argc, char** argv)
{
    std::string filepostfix = ".txt";
    if (argc > 1) {
        filepostfix = argv[1];
    }

    mio::set_log_level(mio::LogLevel::off);

    using namespace mio::mpm;
    using Model  = ABM<QuadWellModel>;
    using Status = Model::Status;

    size_t n_agents = 4 * 300;
    std::vector<double> pop_dist{28. / 30, 2. / 30, 0.0};

    std::list<Model::Agent> agents(n_agents / 4);

    std::vector<ScalarType> pop_qaudrant_1_dist, pop_qaudrant_1 = pop_dist;
    scale(pop_qaudrant_1, n_agents / 4.0);
    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    auto& sta_rng = mio::DiscreteDistribution<int>::get_instance();
    int agent_itr = 0;
    for (auto& a : agents) {
        a.position          = {-1, 1};
        pop_qaudrant_1_dist = pop_qaudrant_1;
        scale(pop_qaudrant_1_dist, 4. / (n_agents - agent_itr));
        a.status = static_cast<Status>(sta_rng(pop_qaudrant_1_dist));
        pop_qaudrant_1[(Eigen::Index)a.status] -= 1;
        agent_itr++;
    }
    // avoid edge cases caused by random starting positions
    // for (auto& agent : agents) {
    //     for (int i = 0; i < 5; i++) {
    //         Model::move(0, 0.002, agent);
    //     }
    // }

    std::vector<AdoptionRate<Status>> adoption_rates = {
        {Status::S, Status::I, 0, 0.3, {Status::I}, {1}}, {Status::I, Status::R, 0, 0.1},
        {Status::S, Status::I, 1, 0.3, {Status::I}, {1}}, {Status::I, Status::R, 1, 0.1},
        {Status::S, Status::I, 2, 0.3, {Status::I}, {1}}, {Status::I, Status::R, 2, 0.1},
        {Status::S, Status::I, 3, 1.0, {Status::I}, {1}}, {Status::I, Status::R, 3, 0.08}};

    Model model(agents, adoption_rates);

    mio::TimeSeries<ScalarType> result(12);

    const unsigned regions = 4;

    SMModel<regions, Status> smm;
    ScalarType kappa                                 = 0.01;
    std::vector<std::vector<ScalarType>> populations = {
        {0, 0, 0}, {n_agents / 4.0, 0, 0}, {n_agents / 4.0, 0, 0}, {n_agents / 4.0, 0, 0}};
    smm.parameters.get<AdoptionRates<Status>>()   = adoption_rates;
    smm.parameters.get<TransitionRates<Status>>() = {
        {Status::S, 0, 1, 0.1 * kappa}, {Status::I, 0, 1, 0.1 * kappa}, {Status::R, 0, 1, 0.1 * kappa},
        {Status::S, 0, 2, 0.1 * kappa}, {Status::I, 0, 2, 0.1 * kappa}, {Status::R, 0, 2, 0.1 * kappa},
        {Status::S, 1, 0, 0.1 * kappa}, {Status::I, 1, 0, 0.1 * kappa}, {Status::R, 1, 0, 0.1 * kappa},
        {Status::S, 1, 3, 0.1 * kappa}, {Status::I, 1, 3, 0.1 * kappa}, {Status::R, 1, 3, 0.1 * kappa},
        {Status::S, 2, 0, 0.1 * kappa}, {Status::I, 2, 0, 0.1 * kappa}, {Status::R, 2, 0, 0.1 * kappa},
        {Status::S, 2, 3, 0.1 * kappa}, {Status::I, 2, 3, 0.1 * kappa}, {Status::R, 2, 3, 0.1 * kappa},
        {Status::S, 3, 1, 0.1 * kappa}, {Status::I, 3, 1, 0.1 * kappa}, {Status::R, 3, 1, 0.1 * kappa},
        {Status::S, 3, 2, 0.1 * kappa}, {Status::I, 3, 2, 0.1 * kappa}, {Status::R, 3, 2, 0.1 * kappa}};

    for (size_t k = 0; k < regions; k++) {
        for (int i = 0; i < static_cast<size_t>(Status::Count); i++) {
            smm.populations[{static_cast<Region>(k), static_cast<Status>(i)}] = populations[k][i];
        }
    }

    PDMModel<regions, Status> pdmm;
    pdmm.parameters.get<AdoptionRates<Status>>()   = smm.parameters.get<AdoptionRates<Status>>();
    pdmm.parameters.get<TransitionRates<Status>>() = smm.parameters.get<TransitionRates<Status>>();
    pdmm.populations                               = smm.populations;

    double delta_time = 0.2;
    double start_time = 0.0;
    double end_time   = 100.0;

    auto simA = mio::Simulation<Model>(model, 0, 0.05);
    auto simB = mio::Simulation<PDMModel<regions, Status>>(pdmm, 0, 0.05);

    TIME_TYPE pre = TIME_NOW;

    for (double t = start_time; t < end_time; t = std::min(t + delta_time, end_time)) {
        printf("%.1f/%.1f\r", t, end_time);
        simA.advance(t);
        simB.advance(t);
        { // abm deduct
            auto& agents = simA.get_model().populations;
            auto itr     = agents.begin();
            while (itr != agents.end()) {
                if (itr->position[0] > 0) {
                    if (itr->position[1] > 0) {
                        simB.get_model().populations[{mio::mpm::Region(2), itr->status}]++;
                    }
                    else {
                        simB.get_model().populations[{mio::mpm::Region(4), itr->status}]++;
                    }
                    itr = agents.erase(itr);
                }
                else if (itr->position[1] < 0) {
                    if (itr->position[0] < 0) {
                        simB.get_model().populations[{mio::mpm::Region(3), itr->status}]++;
                    }
                    else {
                        simB.get_model().populations[{mio::mpm::Region(4), itr->status}]++;
                    }
                    itr = agents.erase(itr);
                }
                itr++;
            }
        }
        { //pdmm/smm deduct
            auto& pop = simB.get_model().populations;
            for (int i = 0; i < (int)Status::Count; i++) {
                for (auto& agents = pop[{mio::mpm::Region(1), (Status)i}]; agents > 0; agents--) {
                    simA.get_model().populations.push_back({{-1, 1}, (Status)i});
                }
            }
        }
    }
    TIME_TYPE post = TIME_NOW;

    {
        std::string fname = "hybrid_lok_top_left_results" + filepostfix;
        FILE* file        = fopen(fname.c_str(), "w");
        print_to_file(file, simA.get_result(),
                      {"S1", "I1", "R1", "S2", "I2", "R2", "S3", "I3", "R3", "S4", "I4", "R4"});
        fprintf(file, "# Elapsed time during advance(): %.*g\n", PRECISION, PRINTABLE_TIME(post - pre));
        fclose(file);
    }

    {
        std::string fname = "hybrid_lok_others_results" + filepostfix;
        FILE* file        = fopen(fname.c_str(), "w");
        print_to_file(file, simB.get_result(),
                      {"S1", "I1", "R1", "S2", "I2", "R2", "S3", "I3", "R3", "S4", "I4", "R4"});
        fprintf(file, "# Elapsed time during advance(): %.*g\n", PRECISION, PRINTABLE_TIME(post - pre));
        fclose(file);
    }

    return 0;
}