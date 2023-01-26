#include "mpm/abm.h"
#include "mpm/utility.h"

#include <map>

enum class InfectionState
{
    S,
    I,
    R,
    Count
};

class DoubleWellModel
{
    constexpr static double contact_radius = 0.6;
    constexpr static double sigma          = 0.4;

public:
    using Status   = InfectionState;
    using Position = Eigen::Vector2d;

    struct Agent {
        Position position;
        InfectionState status;
    };

    DoubleWellModel(const std::vector<Agent>& agents, const typename mio::mpm::AdoptionRates<Status>::Type& rates)
        : populations(agents)
    {
        for (auto& agent : populations) {
            assert(is_in_domain(agent.position));
        }
        for (auto& r : rates) {
            m_adoption_rates[{r.region, r.from, r.to}] = {r.order, r.factor};
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
            if (adoption_rate.first == mio::mpm::Order::first) { // .first is the order
                // contact independant rate
                rate = adoption_rate.second; // .second is the factor
            }
            else { // order == second
                // accumulate rate per contact with status new_status
                size_t num_contacts   = 0;
                size_t num_influences = 0;
                for (auto& contact : populations) {
                    // check if contact is indeed a contact
                    if (is_contact(agent, contact)) {
                        num_contacts++;
                        if (contact.status == new_status) {
                            num_influences++;
                        }
                    }
                }
                // rate = factor * "concentration of contacts with status new_status"
                if (num_contacts > 0) {
                    rate = adoption_rate.second * (static_cast<double>(num_influences) / num_contacts);
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
        Eigen::Matrix<double, 2 * static_cast<size_t>(Status::Count), 1> val;
        val.setZero();
        for (auto& agent : populations) {
            // split population into the wells given by grad_U
            const auto position = (agent.position[0] < 0)
                                      ? static_cast<size_t>(agent.status)
                                      : static_cast<size_t>(agent.status) + static_cast<size_t>(Status::Count);
            val[position]++;
        }
        return val;
    }

    std::vector<Agent> populations;

private:
    static Position grad_U(const Position x)
    {
        // U is a double well potential
        // U(x0,x1) = (x0^2 - 1)^2 + 2*x1^2
        return {4 * x[0] * (x[0] * x[0] - 1), 4 * x[1]};
    }

    bool is_contact(const Agent& agent, const Agent& contact) const
    {
        //      test if contact is in the contact radius                   and test if agent and contact are different objects
        return (agent.position - contact.position).norm() < contact_radius && (&agent != &contact);
    }

    bool is_in_domain(const Position& p) const
    {
        // restrict domain to [-2, 2]^2 where "escaping" is impossible, i.e. it holds x <= grad_U(x)
        return -2 <= p[0] && p[0] <= 2 && -2 <= p[1] && p[1] <= 2;
    }

    std::map<std::tuple<mio::mpm::Region, Status, Status>, std::pair<mio::mpm::Order, double>> m_adoption_rates;
};

int main()
{
    using namespace mio::mpm;
    using Model     = ABM<DoubleWellModel>;
    using Status    = Model::Status;
    size_t n_agents = 2000;

    std::vector<Model::Agent> agents(n_agents);

    std::vector<double> pop_dist{0.95, 0.05, 0.0};
    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    auto& sta_rng = mio::DiscreteDistribution<int>::get_instance();
    for (auto& a : agents) {
        a.position = {pos_rng(-2.0, 2.0), pos_rng(-2.0, 2.0)};
        if (a.position[0] < 0) {
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

    std::vector<AdoptionRate<Status>> adoption_rates = {{Status::S, Status::I, 0.3, Order::second, 0},
                                                        {Status::I, Status::R, 0.1, Order::first, 0},
                                                        {Status::S, Status::I, 1, Order::second, 1},
                                                        {Status::I, Status::R, 0.08, Order::first, 1}};

    Model model(agents, adoption_rates);

    auto result = mio::simulate(0, 100, 0.1, model);

    mio::mpm::print_to_terminal(result, {"S", "I", "R", "S", "I", "R"});

    return 0;
}