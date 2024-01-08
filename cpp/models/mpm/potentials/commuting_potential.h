#ifndef COMMUTING_POTENTIAL_H_
#define COMMUTING_POTENTIAL_H_

#include "hybrid_paper/metaregion_sampler.h"
#include "memilio/math/eigen.h"
#include "memilio/math/floating_point.h"
#include "mpm/model.h"

struct TriangularDistribution {

    TriangularDistribution(double upper_bound, double lower_bound, double mean)
        : mode(std::max(lower_bound, std::min(3 * mean - upper_bound - lower_bound, upper_bound)))
        , upper_bound(upper_bound)
        , lower_bound(lower_bound)
    {
    }
    double mode;
    double upper_bound;
    double lower_bound;

    double get_instance()
    {
        std::array<double, 3> i{lower_bound, mode, upper_bound};
        std::array<double, 3> w{0, 2.0 / (upper_bound - lower_bound), 0};
        double r = std::piecewise_linear_distribution<double>{i.begin(), i.end(), w.begin()}(mio::thread_local_rng());
        assert(r < upper_bound and r > lower_bound);
        return r;
    }
};

struct StochastiK {
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    StochastiK(Eigen::Ref<Matrix> metaregion_commute_weights, Eigen::Ref<const Eigen::MatrixXi> metaregions,
               MetaregionSampler&& metaregion_sampler)
        : metaregion_commute_weights(metaregion_commute_weights)
        , metaregion_sampler(metaregion_sampler)
        , metaregions(metaregions)
    {
    }

    //K
    // if(commutes and (t.isapprox(t_depart) or t.isapprox(t_return)))
    //    x = destination - position
    //    destination = position
    //    return x
    //
    // else
    //    return {0,0}

    template <class Agent>
    Eigen::Vector2d operator()(Agent& a, double t, double dt)
    {
        if (a.commutes and (is_in_interval(a.t_depart, t, t + dt) or is_in_interval(a.t_return, t, t + dt))) {

            Eigen::Vector2d direction = a.commuting_destination - a.position;
            Eigen::Vector2d dest      = a.position + direction;

            assert(metaregions(a.commuting_destination[0], a.commuting_destination[1]) ==
                   metaregions(dest[0], dest[1]));

            a.commuting_destination = a.position;
            return direction;
        }
        else {
            return {0, 0};
        }
    }

    bool is_in_interval(double value, double begin, double end)
    {
        return value >= begin and value < end;
    }

    Matrix metaregion_commute_weights;
    MetaregionSampler metaregion_sampler;
    Eigen::Ref<const Eigen::MatrixXi> metaregions;
};

template <class KProvider, class InfectionState>
class CommutingPotential
{
public:
    using Position       = Eigen::Vector2d;
    using Status         = InfectionState;
    using GradientMatrix = Eigen::Matrix<Eigen::Vector2d, Eigen::Dynamic, Eigen::Dynamic>;

    struct Agent {

        Position position;
        InfectionState status;
        int region;

        bool commutes                  = false;
        Position commuting_destination = {0, 0};
        ScalarType t_return            = 0;
        ScalarType t_depart            = 0;

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

    // KProvider has to implement a function "Position operator() (Agent, double)"
    CommutingPotential(const KProvider& K, const std::vector<Agent>& agents,
                       const typename mio::mpm::AdoptionRates<Status>::Type& rates,
                       Eigen::Ref<const GradientMatrix> potential_gradient,
                       Eigen::Ref<const Eigen::MatrixXi> metaregions,
                       std::vector<InfectionState> non_moving_states = {},
                       const std::vector<double>& sigma              = std::vector<double>(8, 1440.0 / 200.0),
                       const double contact_radius_in_km             = 5)
        : populations(agents)
        , accumulated_contact_rates{0.}
        , contact_rates_count{0}
        , m_k{K}
        , metaregions(metaregions)
        , sigma(sigma)
        , contact_radius(get_contact_radius_factor() * contact_radius_in_km)
        , m_number_transitions(static_cast<size_t>(Status::Count),
                               Eigen::MatrixXd::Zero(metaregions.maxCoeff(), metaregions.maxCoeff()))
        , m_number_commutes(static_cast<size_t>(Status::Count),
                            Eigen::MatrixXd::Zero(metaregions.maxCoeff(), metaregions.maxCoeff()))
        , non_moving_states(non_moving_states)
        , potential_gradient(potential_gradient)

    {
        for (auto& agent : populations) {
            assert(is_in_domain(agent.position));
        }
        for (auto& r : rates) {
            m_adoption_rates.emplace(std::forward_as_tuple(r.region, r.from, r.to), r);
        }
    }

    inline static constexpr void adopt(Agent& agent, const Status& new_status)
    {
        agent.status = new_status;
    }

    double get_total_agents_in_region(int region)
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
                    accumulated_contact_rates += (num_contacts / get_total_agents_in_region(agent.region));
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
            //commuting parameters are drawn at the beginning of every new day
            if (mio::floating_point_equal(t, std::round(t), 1e-10)) {
                draw_commuting_parameters(agent, t, dt);
            }
            Position p = {mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0),
                          mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0)};

            const int region_old = agent.region;
            auto noise           = (sigma[region_old] * std::sqrt(dt)) * p;
            auto new_pos         = agent.position;

            const auto K = m_k(agent, t, dt);
            // TODO: discuss different cases i.e. should we skip gradient when K applies
            if (K != Position{0, 0}) {
                new_pos += K;
            }
            else {
                //use microstepping
                int num_substeps = std::max<int>(noise.norm(), 1);
                for (int substep = 0; substep < num_substeps; ++substep) {
                    new_pos -= (dt * grad_U(agent.position) - noise) / num_substeps;
                }
                //if agents are outside they keep their position
                if (metaregions(new_pos[0], new_pos[1]) == 0) {
                    new_pos = agent.position;
                }
                //if agent crosses the border the influence of the noise term is neglected
                else if (metaregions(new_pos[0], new_pos[1]) - 1 != agent.region) {
                    new_pos -= noise;
                }
            }

            const int region = metaregions(new_pos[0], new_pos[1]) - 1; // shift land so we can use it as index
            if (region >= 0) {
                agent.region = region;
            }
            const bool makes_transition = (region_old != agent.region);

            assert((K != Position{0, 0}) == makes_transition);

            if (makes_transition) {
                m_number_transitions[static_cast<size_t>(agent.status)](region_old, agent.region)++;
                if (m_k.is_in_interval(agent.t_depart, t, t + dt)) {
                    m_number_commutes[static_cast<size_t>(agent.status)](region_old, agent.region)++;
                }
            }
            agent.position = new_pos;
        }
        //else{  agent has a non-moving status   }
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

    double& number_commutes(const mio::mpm::TransitionRate<Status>& tr)
    {
        return m_number_commutes[static_cast<size_t>(tr.status)](static_cast<size_t>(tr.from),
                                                                 static_cast<size_t>(tr.to));
    }

    const std::vector<Eigen::MatrixXd>& number_commutes() const
    {
        return m_number_commutes;
    }

    bool is_contact(const Agent& agent, const Agent& contact) const
    {
        return (&agent != &contact) && // test if agent and contact are different objects
               (agent.region == contact.region) && // test if they are in the same metaregion
               (agent.position - contact.position).norm() < contact_radius; // test if contact is in the contact radius
    }

    std::vector<Agent> populations;
    double accumulated_contact_rates;
    size_t contact_rates_count;
    KProvider m_k;

private:
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

    bool is_in_domain(const Position& p) const
    {
        // restrict domain to [0, num_pixel]^2 where "escaping" is impossible, i.e. it holds x <= grad_U(x)
        return 0 <= p[0] && p[0] <= metaregions.rows() && 0 <= p[1] && p[1] <= metaregions.cols();
    }

    // precomputet discrete gradient
    Position grad_U(const Position p)
    {
        // round each coordinate to the nearest integer
        return potential_gradient(p.x(), p.y());
    }

    void draw_commuting_parameters(Agent& a, const double t, const double dt)
    {
        size_t destination_region =
            mio::DiscreteDistribution<size_t>::get_instance()(m_k.metaregion_commute_weights.row(a.region));

        a.commutes = (destination_region != a.region);

        if (a.commutes) {
            a.commuting_destination = m_k.metaregion_sampler(destination_region);

            assert(metaregions(a.commuting_destination[0], a.commuting_destination[1]) - 1 == destination_region);
            //TODO: anschauen, was Normalverteilung mit den Parametern macht
            //TODO: Zeitmessung triangular dist & normal dist übergeben
            a.t_return = t + mio::ParameterDistributionNormal(9.0 / 24.0, 23.0 / 24.0, 18.0 / 24.0).get_rand_sample();
            a.t_depart = TriangularDistribution(a.t_return - dt, t, t + 9.0 / 24.0).get_instance();

            assert(m_k.is_in_interval(a.t_return, t, t + 1));
            assert(m_k.is_in_interval(a.t_depart, t, t + 1));
            assert(a.t_return < t + 1);
            assert(a.t_depart + dt < a.t_return);
        }
    }

    Eigen::Ref<const Eigen::MatrixXi> metaregions;
    std::map<std::tuple<mio::mpm::Region, Status, Status>, mio::mpm::AdoptionRate<Status>> m_adoption_rates;
    const std::vector<double> sigma;
    const double contact_radius;
    std::vector<Eigen::MatrixXd> m_number_transitions;
    std::vector<Eigen::MatrixXd> m_number_commutes;
    std::vector<InfectionState> non_moving_states;
    Eigen::Ref<const GradientMatrix> potential_gradient;
};

#endif // COMMUTING_POTENTIAL_H_
