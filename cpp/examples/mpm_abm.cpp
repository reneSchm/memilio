#include "mpm/abm.h"
#include "mpm/utility.h"

#include <iostream>
#include <sstream>
#include <fstream>
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

public:
    using Status   = InfectionState;
    using Position = Eigen::Vector2d;

    struct Agent {
        Position position;
        InfectionState status;
        int land;
    };

    DoubleWellModel(const std::vector<Agent>& agents, const typename mio::mpm::AdoptionRates<Status>::Type& rates,
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
            agent.land = land - 1; // shift land so we can use it as index
        }
    }

    Eigen::VectorXd time_point() const
    {
        for (auto agent : populations) {
            std::cout << agent.position[0] << " " << agent.position[1] << " ";
        }
        std::cout << "\n";
        Eigen::Matrix<double, 16 * static_cast<size_t>(Status::Count), 1> val;
        val.setZero();
        for (auto& agent : populations) {
            val[agent.land]++;
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

int main()
{
    using namespace mio::mpm;
    using Model     = ABM<DoubleWellModel>;
    using Status    = Model::Status;
    size_t n_agents = 1;

    std::vector<Model::Agent> agents(n_agents);

    std::vector<double> pop_dist{0.95, 0.05, 0.0};
    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    auto& sta_rng = mio::DiscreteDistribution<int>::get_instance();
    for (auto& a : agents) {
        // a.position = {pos_rng(-2.0, 2.0), pos_rng(-2.0, 2.0)};
        // if (a.position[0] < 0) {
        //     a.status = static_cast<Status>(sta_rng(pop_dist));
        // }
        // else {
        //     a.status = Status::S;
        // }
        a.status   = Status::S;
        a.position = {800, 800};
    }
    // avoid edge cases caused by random starting positions
    // for (auto& agent : agents) {
    //     for (int i = 0; i < 5; i++) {
    //         Model::move(0, 0.1, agent);
    //     }
    // }

    std::vector<AdoptionRate<Status>> adoption_rates;
    for (int i = 0; i < 16; i++) {
        adoption_rates.push_back({Status::S, Status::I, i, 0.3, {Status::I}, {1}});
        adoption_rates.push_back({Status::I, Status::R, i, 0.1});
    }

    Eigen::MatrixXd potential;
    Eigen::MatrixXi metaregions;

    std::cerr << "Setup: Read potential.\n" << std::flush;
    {
        const auto fname = "../../../potentially_germany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            return 1;
        }
        else {
            potential = 8 * read_pgm(ifile);
            ifile.close();
        }
    }
    std::cerr << "Setup: Read metaregions.\n" << std::flush;
    {
        const auto fname = "../../../metagermany.pgm";
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

    Model model(agents, adoption_rates, potential, metaregions);

    std::cerr << "Starting simulation.\n" << std::flush;

    auto result = mio::simulate(0, 100, 0.05, model);

    // mio::mpm::print_to_terminal(result, {"S", "I", "R", "S", "I", "R"});

    return 0;
}