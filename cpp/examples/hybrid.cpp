#include "examples/hybrid.h"
#include "mpm/abm.h"
#include "mpm/model.h"
#include "mpm/smm.h"
#include "mpm/pdmm.h"
#include "mpm/utility.h"

#include <algorithm>
#include <cstddef>
#include <map>

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
    constexpr static double sigma          = 0.6;

public:
    using Status   = InfectionState;
    using Position = Eigen::Vector2d;

    struct Agent {
        Position position;
        InfectionState status;
    };

    QuadWellModel(const std::vector<Agent>& agents, const typename mio::mpm::AdoptionRates<Status>::Type& rates)
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

    std::vector<Agent> populations;

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

template <>
void mio::convert_model(const Simulation<mio::mpm::ABM<QuadWellModel>>&, mio::mpm::PDMModel<4, InfectionState>&)
{
    DEBUG("convert_model abm->pdmm")
}

template <>
void mio::convert_model(const Simulation<mio::mpm::ABM<QuadWellModel>>&, mio::mpm::SMModel<4, InfectionState>&)
{
    DEBUG("convert_model abm->smm")
}

template <>
void mio::convert_model(const Simulation<mio::mpm::PDMModel<4, InfectionState>>& a, mio::mpm::ABM<QuadWellModel>& b)
{
    DEBUG("convert_model pdmmm->abm")
    using Status = InfectionState;

    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    auto values   = a.get_result().get_last_value().eval();

    for (auto& p : b.populations) {
        auto i     = mio::DiscreteDistribution<size_t>::get_instance()(values);
        values[i]  = std::max(0., values[i] - 1);
        p.status   = static_cast<Status>(i % static_cast<size_t>(Status::Count));
        auto well  = i / static_cast<size_t>(Status::Count);
        p.position = {(well & 1) - pos_rng(0., 1.0), pos_rng(0., 1.0) - ((well & 2) >> 1)};
        b.move(0, 0.1, p);
        b.move(0, 0.1, p);
        b.move(0, 0.1, p);
    }
}

template <>
void mio::convert_model(const Simulation<mio::mpm::SMModel<4, InfectionState>>& a, mio::mpm::ABM<QuadWellModel>& b)
{
    DEBUG("convert_model ebm->abm")
    using Status = InfectionState;

    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    auto values   = a.get_result().get_last_value().eval();

    for (auto& p : b.populations) {
        auto i     = mio::DiscreteDistribution<size_t>::get_instance()(values);
        values[i]  = std::max(0., values[i] - 1);
        p.status   = static_cast<Status>(i % static_cast<size_t>(Status::Count));
        auto well  = i / static_cast<size_t>(Status::Count);
        p.position = {(well & 1) - pos_rng(0., 1.0), pos_rng(0., 1.0) - ((well & 2) >> 1)};
        b.move(0, 0.1, p);
        b.move(0, 0.1, p);
        b.move(0, 0.1, p);
    }
}

int main()
{
    using namespace mio::mpm;
    using Model     = ABM<QuadWellModel>;
    using Status    = Model::Status;
    size_t n_agents = 4000;

    std::vector<Model::Agent> agents(n_agents);

    std::vector<double> pop_dist{0.95, 0.05, 0.0};
    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    auto& sta_rng = mio::DiscreteDistribution<int>::get_instance();
    for (auto& a : agents) {
        a.position = {pos_rng(-2.0, 2.0), pos_rng(-2.0, 2.0)};
        if (a.position[0] < 0 && a.position[1] > 0) {
            a.status = static_cast<Status>(sta_rng(pop_dist));
        }
        else {
            a.status = Status::S;
        }
    }
    // avoid edge cases caused by random starting positions
    for (auto& agent : agents) {
        for (int i = 0; i < 5; i++) {
            Model::move(0, 0.1, agent);
        }
    }

    std::vector<AdoptionRate<Status>> adoption_rates = {{Status::S, Status::I, 0, 0.3, {Status::I}, {1}},
                                                        {Status::I, Status::R, 0, 0.1},
                                                        {Status::S, Status::I, 1, 1.0, {Status::I}, {1}},
                                                        {Status::I, Status::R, 1, 0.08}};

    Model abm(agents, adoption_rates);

    //mio::mpm::print_to_terminal(mio::simulate(0, 100, 0.1, model), {"S", "I", "R", "S", "I", "R", "S", "I", "R", "S", "I", "R"});

    const unsigned regions = 4;

    SMModel<regions, Status> smm;
    ScalarType kappa                                 = 0.01;
    std::vector<std::vector<ScalarType>> populations = {{950, 50, 0}, {1000, 0, 0}, {1000, 0, 0}, {1000, 0, 0}};
    smm.parameters.get<AdoptionRates<Status>>()      = adoption_rates;
    smm.parameters.get<TransitionRates<Status>>()    = {
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

    mio::HybridSimulation<mio::mpm::ABM<QuadWellModel>, mio::mpm::PDMModel<regions, Status>> sim(abm, pdmm, 0.5);

    sim.advance(100, [](bool b, const auto& results) {
        const int critical_num_infections = 20;

        bool use_base = true; // some I comps are subcritical
        bool use_sec  = true; // all I comps are critical

        for (size_t i = 0; i < regions; i++) {
            if (results.get_last_value()[mio::flatten_index<mio::Index<Region, Status>>(
                    {Region(i), Status::I}, {Region(regions), Status::Count})] < critical_num_infections) {
                use_base &= true;
                use_sec = false;
            }
            else {
                use_base = false;
                use_sec &= true;
            }
        }

        if (use_base == !use_sec)
            return use_base;
        else
            return b;
    });

    print_to_terminal(sim.get_result(), {"S", "I", "R", "S", "I", "R", "S", "I", "R", "S", "I", "R"});

    return 0;
}