#ifndef QUAD_WELL_H
#define QUAD_WELL_H

#include "infection_state.h"
#include "mpm/model.h"
#include "hybrid_paper/library/infection_state.h"
#include <cmath>

namespace qw
{

using Position = Eigen::Vector2d;

inline size_t well_index(const Position& p)
{
    // 0|1
    // -+-
    // 2|3
    return (p.x() >= 0) + 2 * (p.y() < 0);
}

class MetaregionSampler
{
public:
    MetaregionSampler(const Position& bottom_left, const Position& mid_point, const Position& top_right, double margin)
        : m_ranges(4)
    {
        auto assign_range = [&](Position range_x, Position range_y) {
            auto i = well_index({range_x.sum() / 2, range_y.sum() / 2});
            range_x += Position{margin, -margin};
            range_y += Position{margin, -margin};
            m_ranges[i] = {range_x, range_y};
        };

        assign_range({bottom_left.x(), mid_point.x()}, {mid_point.y(), top_right.y()});
        assign_range({mid_point.x(), top_right.x()}, {mid_point.y(), top_right.y()});
        assign_range({bottom_left.x(), mid_point.x()}, {bottom_left.y(), mid_point.y()});
        assign_range({mid_point.x(), top_right.x()}, {bottom_left.y(), mid_point.y()});
    }

    Position operator()(size_t metaregion_index) const
    {
        const auto& range = m_ranges[metaregion_index];
        return {mio::UniformDistribution<double>::get_instance()(range.first[0], range.first[1]),
                mio::UniformDistribution<double>::get_instance()(range.second[0], range.second[1])};
    }

private:
    // stores pairs of (x-range, y-range)
    std::vector<std::pair<Position, Position>> m_ranges;
};

} // namespace qw

template <class InfectionState>
class QuadWellModel
{

public:
    using Status   = InfectionState;
    using Position = qw::Position;

    struct Agent {
        Position position;
        Status status;
    };

    QuadWellModel(const std::vector<Agent>& agents, const typename mio::mpm::AdoptionRates<Status>::Type& rates,
                  double contact_radius = 0.4, double sigma = 0.4)
        : populations(agents)
        , m_contact_radius(contact_radius)
        , m_sigma(sigma)
        , m_number_transitions(static_cast<size_t>(Status::Count), Eigen::MatrixXd::Zero(4, 4))
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
        const size_t well = qw::well_index(agent.position);
        auto map_itr      = m_adoption_rates.find({well, agent.status, new_status});
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

    void move(const double /*t*/, const double dt, Agent& agent)
    {
        const auto old_well = qw::well_index(agent.position);
        Position p          = {mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0),
                               mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0)};

        agent.position      = agent.position - dt * grad_U(agent.position) + (m_sigma * std::sqrt(dt)) * p;
        const auto new_well = qw::well_index(agent.position);
        if (old_well != new_well) {
            m_number_transitions[static_cast<size_t>(agent.status)](old_well, new_well)++;
            total_transitions(old_well, new_well) += 1;
        }
    }

    Eigen::VectorXd time_point() const
    {
        Eigen::VectorXd val = Eigen::VectorXd::Zero(4 * static_cast<size_t>(Status::Count));
        for (auto& agent : populations) {
            // split population into the wells given by grad_U
            auto position =
                static_cast<size_t>(agent.status) + qw::well_index(agent.position) * static_cast<size_t>(Status::Count);
            val[position] += 1;
        }
        return val;
    }

    double& number_transitions(const mio::mpm::TransitionRate<Status>& tr)
    {
        return m_number_transitions[static_cast<size_t>(tr.status)](static_cast<size_t>(tr.from),
                                                                    static_cast<size_t>(tr.to));
    }

    const std::vector<Eigen::MatrixXd>& number_transitions() const
    {
        return m_number_transitions;
    }

    std::vector<Agent> populations;
    Eigen::Matrix<double, 4, 4> total_transitions = Eigen::Matrix<double, 4, 4>::Zero();

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
    std::vector<Eigen::MatrixXd> m_number_transitions;
};
#endif //QUAD_WELL_H