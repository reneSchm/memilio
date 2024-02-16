#ifndef QUAD_WELL_H
#define QUAD_WELL_H

#include "memilio/math/eigen.h"
#include "hybrid_paper/library/infection_state.h"
#include "mpm/model.h"

class QuadWellModel
{

public:
    using Status   = InfectionState;
    using Position = Eigen::Vector2d;

    struct Agent {
        Position position;
        Status status;
    };

    QuadWellModel(const std::vector<Agent>& agents, const typename mio::mpm::AdoptionRates<Status>::Type& rates,
                  double contact_radius = 0.4, double sigma = 0.4)
        : populations(agents)
        , m_contact_radius(contact_radius)
        , m_sigma(sigma)
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

        agent.position = agent.position - dt * grad_U(agent.position) + (m_sigma * std::sqrt(dt)) * p;
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
        //      test if contact is in the contact radius                     and test if agent and contact are different objects
        return (agent.position - contact.position).norm() < m_contact_radius && (&agent != &contact);
    }

    bool is_in_domain(const Position& p) const
    {
        // restrict domain to [-2, 2]^2 where "escaping" is impossible, i.e. it holds x <= grad_U(x) for dt <= 0.1
        return -2 <= p[0] && p[0] <= 2 && -2 <= p[1] && p[1] <= 2;
    }

    std::map<std::tuple<mio::mpm::Region, Status, Status>, mio::mpm::AdoptionRate<Status>> m_adoption_rates;
    double m_contact_radius;
    double m_sigma;
};
#endif //QUAD_WELL_H