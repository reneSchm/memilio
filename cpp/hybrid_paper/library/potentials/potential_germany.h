#ifndef POTENTIAL_GERMANY_H_
#define POTENTIAL_GERMANY_H_

#include "mpm/utility.h"
#include "mpm/abm.h"

#include <vector>

template <class InfectionState>
class PotentialGermany
{

public:
    using Status   = InfectionState;
    using Position = Eigen::Vector2d;

    struct Agent {
        Position position;
        InfectionState status;
        int region;

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
            obj.add_element("Land", region);
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
                     Eigen::Ref<const Eigen::MatrixXd> potential, Eigen::Ref<const Eigen::MatrixXi> metaregions,
                     std::vector<InfectionState> non_moving_states = {},
                     const std::vector<double>& sigma              = std::vector<double>(8, 1440.0 / 200.0),
                     const double contact_radius_in_km             = 1000000)
        : potential(potential)
        , metaregions(metaregions)
        , populations(agents)
        , sigma(sigma)
        , non_moving_states(non_moving_states)
        , contact_radius(get_contact_radius_factor() * contact_radius_in_km)
        , accumulated_contact_rates{0.}
        , contact_rates_count{0}
        , m_number_transitions(static_cast<size_t>(Status::Count),
                               Eigen::MatrixXd::Zero(metaregions.maxCoeff(), metaregions.maxCoeff()))
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

    double get_total_agents_in_state(int region)
    {
        double num_agents = 0;
        for (auto& agent : populations) {
            num_agents += (double)(agent.region == region);
        }
        return num_agents;
    }

    double adoption_rate(const Agent& agent, const Status& new_status)
    {
        double rate = 0;
        // get the correct adoption rate
        // auto well    = (agent.position[0] < 0) ? 0 : 1;
        auto map_itr = m_adoption_rates.find({agent.region, agent.status, new_status});
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
                    accumulated_contact_rates += (num_contacts / get_total_agents_in_state(agent.region));
                    contact_rates_count += 1;
                }
            }
        }
        // else: no adoption from agent.status to new_status exist
        return rate;
    }

    void move(const double t, const double dt, Agent& agent)
    {
        if (std::find(non_moving_states.begin(), non_moving_states.end(), agent.status) == non_moving_states.end()) {
            Position p = {mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0),
                          mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0)};

            auto region_old  = agent.region;
            auto pnew        = agent.position;
            auto noise       = (sigma[region_old] * std::sqrt(dt)) * p;
            int num_substeps = std::max<int>(noise.norm(), 1);
            for (int substep = 0; substep < num_substeps; ++substep) {
                pnew -= (dt * grad_U(pnew) - noise) / num_substeps;
            }
            if (potential(pnew[0], pnew[1]) < 8) {
                agent.position = pnew;
            }

            const auto region = metaregions(agent.position[0], agent.position[1]);
            if (region > 0) {
                agent.region = region - 1; // shift region so we can use it as index
            }
            const bool makes_transition = (region_old != agent.region);
            if (makes_transition) {
                m_number_transitions[static_cast<size_t>(agent.status)](region_old, agent.region)++;
            }
        }
    }

    Eigen::VectorXd time_point() const
    {
        // metaregions has values from 0 - #regions, where 0 is no particular region (e.g. borders, outside)
        Eigen::VectorXd val = Eigen::VectorXd::Zero(metaregions.maxCoeff() * static_cast<size_t>(Status::Count));

        for (auto& agent : populations) {
            val[(agent.region * static_cast<size_t>(Status::Count) + static_cast<size_t>(agent.status))]++;
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
    double accumulated_contact_rates;
    size_t contact_rates_count;

protected:
    double get_contact_radius_factor(std::vector<double> areas = std::vector<double>{435, 579, 488, 310.7, 667.3, 800,
                                                                                     870.4, 549.3})
    {
        assert(areas.size() == metaregions.maxCoeff());
        std::vector<double> factors(metaregions.maxCoeff());
        //count pixels
        for (size_t metaregion = 1; metaregion <= metaregions.maxCoeff(); ++metaregion) {
            int num = 0;
            for (size_t row = 0; row < metaregions.rows(); ++row) {
                for (size_t pixel = 0; pixel < metaregions.cols(); ++pixel) {
                    if (metaregions(row, pixel) == metaregion) {
                        num += 1;
                        if (pixel > 0 && metaregions(row, pixel - 1) != metaregion) {
                            num += 1;
                        }
                    }
                }
            }
            for (size_t col = 0; col < metaregions.cols(); ++col) {
                for (size_t pixel = 1; pixel < metaregions.rows(); ++pixel) {
                    if (metaregions(pixel, col) == metaregion && metaregions(pixel - 1, col) != metaregion) {
                        num += 1;
                    }
                }
            }
            factors[metaregion - 1] = sqrt(num / areas[metaregion - 1]);
        }
        return std::accumulate(factors.begin(), factors.end(), 0.0) / metaregions.maxCoeff();
    }
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
        size_t x = std::round(p[0]), y = std::round(p[1]); // round each coordinate to the nearest integer
        if (x <= 0 || x >= potential.cols() - 1) {
            std::cout << "x is out of bounds " << std::endl;
        }
        else if (y <= 0 || y >= potential.cols() - 1) {
            std::cout << "y is out of bounds " << std::endl;
        }
        const ScalarType dUdx = (potential(x + 1, y) - potential(x - 1, y)) / 2 * (double)potential.cols() / 50;
        const ScalarType dUdy = (potential(x, y + 1) - potential(x, y - 1)) / 2 * (double)potential.cols() / 50;
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
               (agent.region == contact.region) && // test if they are in the same metaregion
               (agent.position - contact.position).norm() < contact_radius; // test if contact is in the contact radius
    }

    bool is_in_domain(const Position& p) const
    {
        // restrict domain to [-2, 2]^2 where "escaping" is impossible, i.e. it holds x <= grad_U(x)
        return 0 <= p[0] && p[0] <= metaregions.rows() && 0 <= p[1] && p[1] <= metaregions.cols();
    }

    Eigen::Ref<const Eigen::MatrixXd> potential;
    Eigen::Ref<const Eigen::MatrixXi> metaregions;
    std::map<std::tuple<mio::mpm::Region, Status, Status>, mio::mpm::AdoptionRate<Status>> m_adoption_rates;
    const std::vector<double> sigma;
    const double contact_radius;
    std::vector<Eigen::MatrixXd> m_number_transitions;
    std::vector<InfectionState> non_moving_states;
};

template <class InfectionState>
class GradientGermany : public PotentialGermany<InfectionState>
{
public:
    using Base           = PotentialGermany<InfectionState>;
    using GradientMatrix = Eigen::Matrix<Eigen::Vector2d, Eigen::Dynamic, Eigen::Dynamic>;
    using Agent          = typename Base::Agent;
    using Position       = typename Base::Position;
    using Status         = typename Base::Status;

    GradientGermany(const std::vector<Agent>& agents, const typename mio::mpm::AdoptionRates<Status>::Type& rates,
                    Eigen::Ref<const GradientMatrix> potential_gradient, Eigen::Ref<const Eigen::MatrixXi> metaregions,
                    std::vector<InfectionState> non_moving_states = {},
                    const std::vector<double>& sigma              = std::vector<double>(8, 1440.0 / 200.0),
                    const double contact_radius_in_km             = 1000000)
        : PotentialGermany<InfectionState>(agents, rates, Eigen::MatrixXd::Zero(0, 0), metaregions, non_moving_states,
                                           sigma, contact_radius_in_km)
        , potential_gradient(potential_gradient)
    {
    }

    void move(const double t, const double dt, Agent& agent)
    {
        if (std::find(this->non_moving_states.begin(), this->non_moving_states.end(), agent.status) ==
            this->non_moving_states.end()) {
            Position p = {mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0),
                          mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0)};

            const int region_old = agent.region;
            auto noise           = (this->sigma[region_old] * std::sqrt(dt)) * p;
            auto new_pos         = agent.position;

#ifdef USE_MICROSTEPPING // defined in config.h
            int num_substeps = std::max<int>(noise.norm(), 1);
            for (int substep = 0; substep < num_substeps; ++substep) {
                new_pos -= (dt * grad_U(agent.position) - noise) / num_substeps;
            }
#else
            new_pos -= dt * grad_U(agent.position) - noise;
#endif
            agent.position = new_pos;
            const int region =
                this->metaregions(agent.position[0], agent.position[1]) - 1; // shift region so we can use it as index
            if (region >= 0) {
                agent.region = region;
            }
            const bool makes_transition = (region_old != agent.region);
            if (makes_transition) {
                this->m_number_transitions[static_cast<size_t>(agent.status)](region_old, agent.region)++;
            }
        }
    }

protected:
    // precomputet discrete gradient
    Position grad_U(const Position p)
    {
        // round each coordinate to the nearest integer
        return potential_gradient(p.x(), p.y());
    }

    Eigen::Ref<const GradientMatrix> potential_gradient;
};

#endif // POTENTIAL_GERMANY_H_
