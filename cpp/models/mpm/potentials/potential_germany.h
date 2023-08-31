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
        , contact_radius((double)potential.cols())
        , accumulated_contact_rates{0.}
        , contact_rates_count{0}
        , m_number_transitions(static_cast<size_t>(Status::Count), Eigen::Matrix<double, 16, 16>::Zero())
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

    double get_total_agents_in_state(int land)
    {
        double num_agents = 0;
        for (auto& agent : populations) {
            num_agents += (double)(agent.land == land);
        }
        return num_agents;
    }

    double adoption_rate(const Agent& agent, const Status& new_status)
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
                    accumulated_contact_rates += (num_contacts / get_total_agents_in_state(agent.land));
                    contact_rates_count += 1;
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

        auto land_old    = agent.land;
        auto pnew        = agent.position;
        auto noise       = (sigma * std::sqrt(dt)) * p;
        int num_substeps = std::max<int>(noise.norm(), 1);
        for (int substep = 0; substep < num_substeps; ++substep) {
            pnew -= (dt * grad_U(pnew) - noise) / num_substeps;
        }
        if (potential(pnew[0], pnew[1]) < 8) {
            agent.position = pnew;
        }
        // auto pos = agent.position;
        // for (int k = 0; k < noise.norm(); k++) {
        //     pos = +k / noise.norm() * noise;
        //     agent.position += dt * grad_U(pos) + 1 / noise.norm() * noise;
        // }
        // agent.position = agent.position - dt * grad + noise; //grad_U(agent.position)
        // if (agent.position[0] < 0 || agent.position[1] < 0) {
        //     std::cout << "postion<0\n" << std::flush;
        //     std::cout << "land old: " << land_old << std::endl;
        //     std::cout << "x " << agent.position[0] << "y " << agent.position[1] << std::endl;
        //     std::cout << "x_old" << pold[0] << "y_old" << pold[1] << std::endl;
        //     std::cout << "grad " << grad_U(pold) << std::endl;
        //     std::cout << "sigma " << sigma << std::endl;
        //     std::cout << "p[0]:" << p[0] << "p[1]: " << p[1] << std::endl;
        //     agent.position[0] = std::max(0.0, agent.position[0]);
        //     agent.position[1] = std::max(0.0, agent.position[1]);
        //     std::cout << "maximizing worked" << std::endl;
        //     // const auto land = metaregions(agent.position[0], agent.position[1]);
        //     // std::cout << "worked" << std::endl;
        // }
        // if (agent.position[0] > double(potential.rows() - 1) || agent.position[1] > double(potential.cols() - 1)) {
        //     std::cout << "position>cols\n" << std::flush;
        //     std::cout << "land old: " << land_old << std::endl;
        //     std::cout << "x " << agent.position[0] << "y " << agent.position[1] << std::endl;
        //     std::cout << "x_old" << pold[0] << "y_old" << pold[1] << std::endl;
        //     std::cout << "grad " << grad_U(pold) << std::endl;
        //     std::cout << "sigma " << sigma << std::endl;
        //     std::cout << "p[0]:" << p[0] << "p[1]: " << p[1] << std::endl;
        //     agent.position[0] = std::min((double)potential.rows() - 1, agent.position[0]);
        //     agent.position[1] = std::min((double)potential.cols() - 1, agent.position[1]);
        //     std::cout << "minimizing worked" << std::endl;
        //     // const auto land = metaregions(agent.position[0], agent.position[1]);
        //     // std::cout << "worked" << std::endl;
        // }
        // if (potential(pold[0], pold[1]) == 0 && potential(agent.position[0], agent.position[1]) == 8) {
        //     std::cout << "agent jumps over border!" << std::endl;
        // }
        const auto land = metaregions(agent.position[0], agent.position[1]);
        if (land > 0) {
            agent.land = land - 1;
        }
        const bool makes_transition = (land_old != agent.land);
        if (makes_transition) {
            //std::cout << "num_transitions" << std::endl;
            m_number_transitions[static_cast<size_t>(agent.status)](land_old, agent.land)++;
            //std::cout << "num transitions done" << std::endl;
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

    double& number_transitions(const mio::mpm::TransitionRate<Status>& tr)
    {
        return m_number_transitions[static_cast<size_t>(tr.status)](static_cast<size_t>(tr.from),
                                                                    static_cast<size_t>(tr.to));
    }

    std::vector<Agent> populations;
    double accumulated_contact_rates;
    size_t contact_rates_count;

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
    const double sigma, contact_radius;
    std::vector<Eigen::MatrixXd> m_number_transitions;
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
#endif // POTENTIAL_GERMANY_H_