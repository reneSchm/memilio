#include "hybrid_paper/infection_state.h"
#include "hybrid_paper/initialization.h"
#include "hybrid_paper/weighted_gradient.h"
#include "memilio/config.h"
#include "memilio/math/eigen.h"
#include "memilio/math/floating_point.h"
#include "memilio/utils/random_number_generator.h"
#include "mpm/potentials/potential_germany.h"

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <utility>
#include <vector>

template <class KProvider, class InfectionState>
class KModel : public GradientGermany<InfectionState>
{
public:
    using Base           = GradientGermany<InfectionState>;
    using Position       = typename Base::Position;
    using Status         = typename Base::Status;
    using GradientMatrix = typename Base::GradientMatrix;

    using Agent = typename Base::Agent;
    // struct Agent : public Base::Agent {
    //     Agent(const typename Base::Agent& a)
    //         : Base::Agent(a)
    //     {
    //     }

    //     Agent(const Position& p = {0, 0}, const Status& s = (InfectionState)0, const int& l = 0,
    //           const Position& h = {0, 0}, const Position& c = {0, 0}, const bool& b = true)
    //         : Base::Agent{p, s, l}
    //         , home(h)
    //         , commute(c)
    //         , at_home(b)
    //     {
    //     }

    //     operator typename Base::Agent() &
    //     {
    //         return *this;
    //     }

    //     // Position position;
    //     // Status status;
    //     // int land;

    //     Position home, commute;
    //     bool at_home = true;

    //     /**
    //      * serialize agents.
    //      * @see mio::serialize
    //      */
    //     template <class IOContext>
    //     void serialize(IOContext& io) const
    //     {
    //         auto obj = io.create_object("Agent");
    //         obj.add_element("Position", this->position);
    //         obj.add_element("Status", this->status);
    //         obj.add_element("Land", this->land);
    //     }

    //     /**
    //      * deserialize an object of this class.
    //      * @see mio::deserialize
    //      */
    //     template <class IOContext>
    //     static mio::IOResult<Agent> deserialize(IOContext& io)
    //     {
    //         auto obj   = io.expect_object("Agent");
    //         auto pos   = obj.expect_element("Position", mio::Tag<Position>{});
    //         auto state = obj.expect_element("Status", mio::Tag<InfectionState>{});
    //         auto land  = obj.expect_element("Land", mio::Tag<int>{});
    //         return apply(
    //             io,
    //             [](auto&& pos_, auto&& state_, auto&& land_) {
    //                 return Agent{pos_, state_, land_};
    //             },
    //             pos, state, land);
    //     }
    // };

    // KProvider has to implement a function "Position operator() (Agent, double)"
    KModel(const KProvider& K, const std::vector<Agent>& agents,
           const typename mio::mpm::AdoptionRates<Status>::Type& rates,
           Eigen::Ref<const GradientMatrix> potential_gradient, Eigen::Ref<const Eigen::MatrixXi> metaregions,
           std::vector<InfectionState> non_moving_states = {},
           const std::vector<double>& sigma              = std::vector<double>(8, 1440.0 / 200.0),
           const double contact_radius_in_km             = 100)
        : Base(agents, rates, potential_gradient, metaregions, non_moving_states, sigma, contact_radius_in_km)
        , m_k(K)
    {
    }

    void move(const double t, const double dt, Agent& agent)
    {
        // if (std::find(this->non_moving_states.begin(), this->non_moving_states.end(), agent.status) ==
        //     this->non_moving_states.end()) {
        Position p = {mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0),
                      mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0)};

        const int land_old = agent.land;
        auto noise         = (this->sigma[land_old] * std::sqrt(dt)) * p;
        auto new_pos       = agent.position;

        // #ifdef USE_MICROSTEPPING // defined in config.h
        //             int num_substeps = std::max<int>(noise.norm(), 1);
        //             for (int substep = 0; substep < num_substeps; ++substep) {
        //                 new_pos -= (dt * grad_U(agent.position) - noise) / num_substeps;
        //             }
        // #else
        const auto K = m_k(agent, t);
        new_pos += -dt * this->grad_U(agent.position) + noise + K;
        // #endif
        agent.position = new_pos;
        const int land =
            this->metaregions(agent.position[0], agent.position[1]) - 1; // shift land so we can use it as index
        if (land >= 0) {
            agent.land = land;
        }
        const bool makes_transition = (land_old != agent.land);
        assert((K != Position{0, 0}) == makes_transition);
        if (makes_transition) {
            this->m_number_transitions[static_cast<size_t>(agent.status)](land_old, agent.land)++;
        }
        // }
    }

private:
    KProvider m_k;
};

class MetaregionSampler
{
public:
    MetaregionSampler(Eigen::Ref<const Eigen::MatrixXi> metaregions)
        : m_metaregion_positions(metaregions.maxCoeff())
    {
        for (Eigen::Index i = 0; i < metaregions.rows(); i++) {
            for (Eigen::Index j = 0; j < metaregions.cols(); j++) {
                if (metaregions(i, j) > 0) {
                    m_metaregion_positions[metaregions(i, j) - 1].emplace_back(i, j);
                }
            }
        }
    }

    Eigen::Vector2d operator()(size_t metaregion_index)
    {
        // TODO: discuss implementation: Which positions should be eligible? What distribution do we choose?
        // Suggestion: pick only points from within the circle of maximum radius fitting into a region.
        size_t position_index =
            mio::UniformIntDistribution<size_t>::get_instance()(m_metaregion_positions[metaregion_index].size());
        return m_metaregion_positions[metaregion_index][position_index];
    }

private:
    std::vector<std::vector<Eigen::Vector2d>> m_metaregion_positions;
};

bool should_commute_now(double t)
{
    // TODO: Discuss and validate timing check
    return mio::floating_point_equal(t, std::round(t));
}

struct StochastiK {
    StochastiK(Eigen::Ref<Eigen::MatrixXd> metaregion_commute_weights, Eigen::Ref<const Eigen::MatrixXi> metaregions)
        : metaregion_commute_weights(metaregion_commute_weights)
        , metaregion_sampler(metaregions)
    {
    }

    template <class Agent>
    Eigen::Vector2d operator()(Agent a, double t)
    {
        if (should_commute_now(t)) {
            // TODO: Validate this vector
            return metaregion_sampler(
                       mio::DiscreteDistribution<size_t>::get_instance()(metaregion_commute_weights.row(a.land))) -
                   a.position;
        }
        else {
            return {0, 0};
        }
    }

    Eigen::MatrixXd metaregion_commute_weights;
    MetaregionSampler metaregion_sampler;
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
    std::string path = "../../";
    using KProvider  = StochastiK;
    using Model      = mio::mpm::ABM<KModel<KProvider, mio::mpm::paper::InfectionState>>;

    auto weights = std::vector<double>(14, 1000);
    auto sigmas  = std::vector<double>(8, 10);
    // double slope = 0; unused

    WeightedGradient wg(path + "potentially_germany_grad.json", path + "boundary_ids.pgm");

    std::string agent_file = "/group/HPC/Gruppen/PSS/Modelle/Hybrid Models/Papers, Theses, "
                             "Posters/2023_Paper_Spatio-temporal_hybrid/initializations/initSusceptible9375.json";
    std::vector<Model::Agent> agents;
    read_initialization(agent_file, agents);

    wg.apply_weights(weights);

    auto metaregions_pgm_file = path + "metagermany.pgm";
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
    Eigen::MatrixXd reference_commuters = get_transition_matrix(path + "data/mobility/").value();
    auto ref_pops = std::vector<double>{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    auto ref_pop  = std::accumulate(ref_pops.begin(), ref_pops.end(), 0.0);

    Eigen::MatrixXd commute_weights(8, 8);
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            if (i == j) {
                commute_weights(i, i) = ref_pops[i] - reference_commuters.row(i).sum();
            }
            else {
                commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
            }
        }
    }

    // TODO: adjust weights to daily commuter rates
    StochastiK k_provider(commute_weights, metaregions);
    //  DeterministiK k_provider(); // TODO: have to set home, commute, at_home for each agent
    Model m(k_provider, agents, {}, wg.gradient, metaregions, {}, sigmas);

    double tmax = 100;

    mio::Simulation<Model> sim(m, 0, 0.1);
    sim.advance(tmax);
    auto results = sim.get_result();

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
                                          {Model::Status::S, mio::mpm::Region(i), mio::mpm::Region(j)}) /
                                          sim.get_model().populations.size(),
                                      reference_commuters(county_ids[i], county_ids[j]) * tmax / ref_pop)
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

    auto file = fopen((path + "output.txt").c_str(), "w");
    mio::mpm::print_to_file(file, results, {});
    fclose(file);

    return 0;
}