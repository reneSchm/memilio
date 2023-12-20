#include "abm/time.h"
#include "hybrid_paper/infection_state.h"
#include "hybrid_paper/initialization.h"
#include "hybrid_paper/weighted_gradient.h"
#include "memilio/config.h"
#include "memilio/math/eigen.h"
#include "memilio/math/floating_point.h"
#include "memilio/utils/parameter_distributions.h"
#include "memilio/utils/random_number_generator.h"
#include "memilio/utils/span.h"
#include "mpm/potentials/potential_germany.h"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

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

template <class KProvider, class InfectionState>
class KModel
{
public:
    using Position       = Eigen::Vector2d;
    using Status         = InfectionState;
    using GradientMatrix = Eigen::Matrix<Eigen::Vector2d, Eigen::Dynamic, Eigen::Dynamic>;

    struct Agent {

        Position position;
        InfectionState status;
        int land;

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

    //Agent:(Base::Agent)
    // - bool commutes
    // - Position Destination (Origin)
    // - timepoint t_return
    // - timepoint t_depart

    //move:
    // if(t is integer)
    //       draw commuting stuff
    // move() with k
    //

    //K
    // if(commutes and (t.isapprox(t_depart) or t.isapprox(t_return)))
    //    x = destination - position
    //    destination = position
    //    return x
    //
    // else
    //    return {0,0}

    // KProvider has to implement a function "Position operator() (Agent, double)"
    KModel(const KProvider& K, const std::vector<Agent>& agents,
           const typename mio::mpm::AdoptionRates<Status>::Type& rates,
           Eigen::Ref<const GradientMatrix> potential_gradient, Eigen::Ref<const Eigen::MatrixXi> metaregions,
           std::vector<InfectionState> non_moving_states = {},
           const std::vector<double>& sigma              = std::vector<double>(8, 1440.0 / 200.0),
           const double contact_radius_in_km             = 100)
        : populations(agents)
        , accumulated_contact_rates{0.}
        , contact_rates_count{0}
        , metaregions(metaregions)
        , sigma(sigma)
        , contact_radius(get_contact_radius_factor() * contact_radius_in_km)
        , m_number_transitions(static_cast<size_t>(Status::Count),
                               Eigen::MatrixXd::Zero(metaregions.maxCoeff(), metaregions.maxCoeff()))
        , non_moving_states(non_moving_states)
        , potential_gradient(potential_gradient)
        , m_k{K}
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

        if (std::find(non_moving_states.begin(), non_moving_states.end(), agent.status) == non_moving_states.end()) {
            //check whether commuting parameters should be drawn
            if (mio::floating_point_equal(t, std::round(t))) {
                draw_commuting_parameters(agent, t, dt);
            }
            Position p = {mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0),
                          mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0)};

            const int land_old = agent.land;
            auto noise         = (sigma[land_old] * std::sqrt(dt)) * p;
            auto new_pos       = agent.position;

            // #ifdef USE_MICROSTEPPING // defined in config.h
            // int num_substeps = std::max<int>(noise.norm(), 1);
            // for (int substep = 0; substep < num_substeps; ++substep) {
            //     new_pos -= (dt * grad_U(agent.position) - noise) / num_substeps;
            // }
            // #else
            const auto K = m_k(agent, t, dt);
            new_pos += -dt * grad_U(agent.position) + noise + K;
            if (K != Position{0, 0} and agent.land == 0) {
                Position grad = grad_U(agent.position);
                //assert(grad_U(agent.position) == Position::Zero(2));
                size_t dest = metaregions(new_pos[0], new_pos[1]) - 1;
                matrix(1, dest)++;
            }
            // #endif
            agent.position = new_pos;
            const int land =
                metaregions(agent.position[0], agent.position[1]) - 1; // shift land so we can use it as index
            if (land >= 0) {
                agent.land = land;
            }
            const bool makes_transition = (land_old != agent.land);
            assert((K != Position{0, 0}) == makes_transition);
            if (makes_transition) {
                m_number_transitions[static_cast<size_t>(agent.status)](land_old, agent.land)++;
            }
        }
        //else{  agent has a non-moving status   }
    }
    Eigen::VectorXd time_point() const
    {
        // metaregions has values from 0 - #regions, where 0 is no particular region (e.g. borders, outside)
        Eigen::VectorXd val = Eigen::VectorXd::Zero(metaregions.maxCoeff() * static_cast<size_t>(Status::Count));

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

    const std::vector<Eigen::MatrixXd>& number_transitions() const
    {
        return m_number_transitions;
    }

    std::vector<Agent> populations;
    double accumulated_contact_rates;
    size_t contact_rates_count;

    Eigen::MatrixXi matrix = Eigen::MatrixXi::Zero(2, 8);

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
    bool is_contact(const Agent& agent, const Agent& contact) const
    {
        return (&agent != &contact) && // test if agent and contact are different objects
               (agent.land == contact.land) && // test if they are in the same metaregion
               (agent.position - contact.position).norm() < contact_radius; // test if contact is in the contact radius
    }

    bool is_in_domain(const Position& p) const
    {
        // restrict domain to [-2, 2]^2 where "escaping" is impossible, i.e. it holds x <= grad_U(x)
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
            mio::DiscreteDistribution<size_t>::get_instance()(m_k.metaregion_commute_weights.row(a.land));

        a.commutes = destination_region != a.land;
        if (a.land == 0) {
            matrix(0, destination_region)++;
        }
        if (a.commutes) {
            a.commuting_destination = m_k.metaregion_sampler(destination_region);
            assert(metaregions(a.commuting_destination[0], a.commuting_destination[1]) - 1 == destination_region);
            a.t_return = t + mio::ParameterDistributionNormal(9.0 / 24.0, 23.0 / 24.0, 18.0 / 24.0).get_rand_sample();
            a.t_depart = TriangularDistribution(a.t_return - dt, t, t + 9.0 / 24.0).get_instance();
            assert(m_k.is_in_interval(a.t_return, t, t + 1));
            assert(m_k.is_in_interval(a.t_depart, t, t + 1));
            assert(a.t_return < t + 1);
            assert(a.t_depart + 0.1 < a.t_return);
        }
    }

    Eigen::Ref<const Eigen::MatrixXi> metaregions;
    std::map<std::tuple<mio::mpm::Region, Status, Status>, mio::mpm::AdoptionRate<Status>> m_adoption_rates;
    const std::vector<double> sigma;
    const double contact_radius;
    std::vector<Eigen::MatrixXd> m_number_transitions;
    std::vector<InfectionState> non_moving_states;
    Eigen::Ref<const GradientMatrix> potential_gradient;

public:
    KProvider m_k;
};

class MetaregionSampler
{
public:
    MetaregionSampler(Eigen::Ref<const Eigen::MatrixXi> metaregions, Eigen::Ref<const Eigen::MatrixXi> boundary_ids)
        : m_metaregion_positions(metaregions.maxCoeff())
    {
        Eigen::MatrixXi is_outside = (1 - metaregions.cast<bool>().array()).matrix().cast<int>();
        extend_bitmap(is_outside, 2);
        for (Eigen::Index i = 0; i < metaregions.rows(); i++) {
            for (Eigen::Index j = 0; j < metaregions.cols(); j++) {
                if (metaregions(i, j) > 0 && !is_outside(i, j)) {
                    m_metaregion_positions[metaregions(i, j) - 1].emplace_back(i, j);
                }
            }
        }
    }

    void extend_bitmap(Eigen::Ref<Eigen::MatrixXi> bitmap, Eigen::Index width)
    {
        Eigen::Index extent    = 2 * width + 1;
        Eigen::MatrixXi canvas = Eigen::MatrixXi::Zero(bitmap.rows() + extent, bitmap.cols() + extent);
        for (Eigen::Index i = 0; i < bitmap.rows(); i++) {
            for (Eigen::Index j = 0; j < bitmap.cols(); j++) {
                for (Eigen::Index k = 0; k < extent; k++) {
                    for (Eigen::Index l = 0; l < extent; l++) {
                        canvas(i + k, j + l) |= bitmap(i, j);
                    }
                }
            }
        }
        bitmap = canvas.block(width, width, bitmap.rows(), bitmap.cols());
    }

    Eigen::Vector2d operator()(size_t metaregion_index)
    {
        // TODO: discuss implementation: Which positions should be eligible? What distribution do we choose?
        // Suggestion: pick only points from within the circle of maximum radius fitting into a region.
        size_t position_index =
            mio::UniformIntDistribution<size_t>::get_instance()(0, m_metaregion_positions[metaregion_index].size());
        return m_metaregion_positions[metaregion_index][position_index];
    }

private:
    std::vector<std::vector<Eigen::Vector2d>> m_metaregion_positions;
};

bool should_commute_now(double t)
{
    //draw if
    // TODO: Discuss and validate timing check
    return mio::floating_point_equal(t, std::round(t));
}

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
    int comm    = 0;
    int departs = 0;

    template <class Agent>
    Eigen::Vector2d operator()(Agent& a, double t, double dt)
    {
        if (a.land == 0 and a.commutes) {
            comm++;
            assert(a.t_depart < 1);
            if (is_in_interval(a.t_depart, t, t + dt)) {
                ++departs;
            }
            else {
                std::cout << "t_depart: " << a.t_depart << " t: " << t << " dt:" << dt << "\n";
            }
        }
        if (a.commutes and (is_in_interval(a.t_depart, t, t + dt) or is_in_interval(a.t_return, t, t + dt))) {
            Eigen::Vector2d direction = a.commuting_destination - a.position;

            Eigen::Vector2d dest = a.position + direction;
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

struct DeterministiK {
    template <class Agent>
    Eigen::Vector2d operator()(Agent a, double t)
    {
        if (should_commute_now(t)) {
            if (a.at_home) {
                return a.commute - a.position;
            }
            else {
                return a.home - a.position;
            }
        }
        else {
            return {0, 0};
        }
    }
};

std::string colorize(double a, double b)
{
    std::stringstream ss("");
    if (a / b < 1) {
        double proc = 1 - a / b;
        if (proc <= 0.25) {
            ss << "\033[32m"; // green
        }
        else if (proc <= 0.50) {
            ss << "\033[33m"; // yellow
        }
        else {
            ss << "\033[31m"; // red
        }
    }
    else {
        double proc = 1 - b / a;
        if (proc <= 0.25) {
            ss << "\033[42m"; // green
        }
        else if (proc <= 0.50) {
            ss << "\033[43m"; // yellow
        }
        else {
            ss << "\033[41m"; // red
        }
    }
    ss << a << " / " << b << "\033[0m";
    return ss.str();
}

int main(int argc, char** argv)
{
    using KProvider = StochastiK;
    using Model     = mio::mpm::ABM<KModel<KProvider, mio::mpm::paper::InfectionState>>;

    auto weights = std::vector<double>(14, 0);
    auto sigmas  = std::vector<double>(8, 0);
    // double slope = 0; unused

    WeightedGradient wg(mio::base_dir() + "potentially_germany_grad.json", mio::base_dir() + "boundary_ids.pgm");

    std::string agent_file = "/group/HPC/Gruppen/PSS/Modelle/Hybrid Models/Papers, Theses, "
                             "Posters/2023_Paper_Spatio-temporal_hybrid/initializations/initSusceptible9375.json";
    std::vector<Model::Agent> agents;
    read_initialization(agent_file, agents);

    wg.apply_weights(weights);

    auto metaregions_pgm_file = mio::base_dir() + "metagermany.pgm";
    auto metaregions          = [&metaregions_pgm_file]() {
        std::ifstream ifile(metaregions_pgm_file);
        if (!ifile.is_open()) { // write error and abort
            mio::log(mio::LogLevel::critical, "Could not open metaregion file {}", metaregions_pgm_file);
            exit(1);
        }
        else { // read pgm from file and return matrix
            auto tmp = mio::mpm::read_pgm_raw(ifile).first;
            ifile.close();
            return tmp;
        }
    }();

    const std::vector<int> county_ids   = {233, 228, 242, 238, 223, 232, 231, 229};
    Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();
    auto ref_pops = std::vector<double>{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    auto ref_pop  = std::accumulate(ref_pops.begin(), ref_pops.end(), 0.0);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(8, 8);
    commute_weights.setZero();
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i != j) {
                commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
            }
        }
        commute_weights(i, i) = ref_pops[i] - commute_weights.row(i).sum();
    }
    std::cout << "commute_weights: ";
    std::cout << commute_weights.row(0).sum() << "\n";
    std::cout << commute_weights << "\n";

    // TODO: adjust weights to daily commuter rates
    StochastiK k_provider(commute_weights, metaregions, {metaregions, wg.boundary_ids});
    Eigen::VectorXi v = Eigen::VectorXi::Zero(8);
    std::cout << "probs: "
              << k_provider.metaregion_commute_weights.row(0) / k_provider.metaregion_commute_weights.row(0).sum()
              << "\n";
    std::vector<double> vv(k_provider.metaregion_commute_weights.row(0).data(),
                           k_provider.metaregion_commute_weights.row(0).data() +
                               k_provider.metaregion_commute_weights.row(0).size());
    std::cout << k_provider.metaregion_commute_weights.row(0) << "\n";
    for (auto value : vv) {
        std::cout << value << " ";
    }
    std::cout << "\n";
    for (int n = 0; n < 100000; ++n) {
        v(mio::DiscreteDistribution<size_t>::get_instance()(vv))++;
    }
    std::cout << "v " << v.transpose() << "\n";
    //  DeterministiK k_provider(); // TODO: have to set home, commute, at_home for each agent
    Model m(k_provider, agents, {}, wg.gradient, metaregions, {}, sigmas);

    double tmax = 1;

    mio::Simulation<Model> sim(m, 0, 0.1);
    sim.advance(tmax);
    auto results = sim.get_result();
    std::cout << "check\n";
    std::cout << sim.get_model().matrix << "\n";
    std::cout << sim.get_model().m_k.comm << "\n";
    std::cout << sim.get_model().m_k.departs << "\n";

    double flows[8] = {0, 0, 0, 0, 0, 0, 0, 0};

    std::cout << "#############\n";

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i != j &&
                sim.get_model().number_transitions({Model::Status::S, mio::mpm::Region(i), mio::mpm::Region(j)})) {
                flows[i] -=
                    sim.get_model().number_transitions({Model::Status::S, mio::mpm::Region(i), mio::mpm::Region(j)});
                flows[i] +=
                    sim.get_model().number_transitions({Model::Status::S, mio::mpm::Region(j), mio::mpm::Region(i)});

                std::cout << i << "->" << j << " : "
                          << (reference_commuters(county_ids[i], county_ids[j]) * tmax / ref_pop > 0.5 ? "#" : " ")
                          << colorize(sim.get_model().number_transitions(
                                          {Model::Status::S, mio::mpm::Region(i), mio::mpm::Region(j)}), //
                                      (reference_commuters(county_ids[i], county_ids[j]) +
                                       reference_commuters(county_ids[j], county_ids[i])) *
                                          tmax * sim.get_model().populations.size() / ref_pop) //
                          << "\n";
            }
        }
    }

    int count = 0;
    for (auto a : sim.get_model().populations) {
        if (metaregions(a.position[0], a.position[1]) == 0) {
            count++;
        }
    }

    std::cout << "#############\nleft out : " << count << "\n";

    std::cout << "flows:\n";
    for (int i = 0; i < 8; i++) {
        std::cout << i << ": " << flows[i] << "\n";
    }

    std::cout << "\n" << sim.get_model().number_transitions()[0] << "\n";

    auto file = fopen((mio::base_dir() + "output.txt").c_str(), "w");
    mio::mpm::print_to_file(file, results, {});
    fclose(file);

    return 0;
}