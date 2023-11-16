#include "memilio/data/analyze_result.h"
#include "hybrid_paper/weighted_gradient.h"
#include "mpm/abm.h"
#include "hybrid_paper/infection_state.h"
#include "memilio/io/json_serializer.h"
#include "mpm/potentials/potential_germany.h"

namespace mio
{
namespace mpm
{
namespace paper
{

void get_agent_movement(size_t n_agents, Eigen::MatrixXi& metaregions, Eigen::MatrixXd& potential, WeightedGradient& wg)
{
    using ABM = mio::mpm::ABM<GradientGermany<InfectionState>>;
    std::vector<ABM::Agent> agents(n_agents);
    auto& pos_rng = mio::UniformDistribution<double>::get_instance();
    for (auto& a : agents) {
        Eigen::Vector2d pos_candidate{pos_rng(0.0, double(potential.rows())), pos_rng(0.0, double(potential.cols()))};
        while (metaregions(pos_candidate[0], pos_candidate[1]) == 0 ||
               potential(pos_candidate[0], pos_candidate[1]) != 0) {
            pos_candidate = {pos_rng(0.0, double(potential.rows())), pos_rng(0.0, double(potential.cols()))};
        }
        a.position = {pos_candidate[0], pos_candidate[1]};
        a.land     = metaregions(pos_candidate[0], pos_candidate[1]) - 1;
        a.status   = InfectionState::S;
    }

    ABM model(agents, {}, wg.gradient, metaregions);
    const double dt   = 0.05;
    double t          = 0;
    const double tmax = 100;

    auto sim = mio::Simulation<ABM>(model, t, dt);
    while (t < tmax) {
        for (auto& agent : sim.get_model().populations) {
            std::cout << agent.position[0] << " " << agent.position[1] << " ";
            sim.get_model().move(t, dt, agent);
        }
        std::cout << "\n";
        t += dt;
    }
}

template <class Agent>
void read_initialization(std::string filename, std::vector<Agent>& agents)
{
    auto result = mio::read_json(filename).value();
    for (int i = 0; i < result.size(); ++i) {
        auto a = mio::deserialize_json(result[std::to_string(i)], mio::Tag<Agent>{}).value();
        agents.push_back(Agent{a.position, a.status, a.land});
    }
}

void run_multiple_simulation(std::string init_file, std::vector<mio::mpm::AdoptionRate<InfectionState>>& adoption_rates,
                             WeightedGradient& wg, const std::vector<double>& sigma, Eigen::MatrixXi& metaregions, double tmax, double delta_t,
                             int num_runs)
{
    using ABM = mio::mpm::ABM<GradientGermany<InfectionState>>;
    std::vector<ABM::Agent> agents;
    read_initialization<ABM::Agent>(init_file, agents);

    int num_agents = agents.size();

    size_t num_regions = metaregions.maxCoeff();

    std::vector<mio::TimeSeries<double>> ensemble_results(
        num_runs, mio::TimeSeries<double>::zero(tmax, num_regions * static_cast<size_t>(InfectionState::Count)));
    for(int run = 0; run < num_runs; ++run){
        std::cerr << "run number: " << run << "\n" << std::flush;
        std::vector<ABM::Agent> agents_run = agents;
        ABM model(agents, adoption_rates, wg.gradient, metaregions, {InfectionState::D}, sigma);
        auto run_result = mio::simulate(0.0, tmax, delta_t, model);
        ensemble_results[run] = mio::interpolate_simulation_result(run_result);
    }
}

} //namespace paper
} //namespace mpm
} //namespace mio

int main()
{
    Eigen::MatrixXi metaregions;
    Eigen::MatrixXd potential;

    WeightedGradient wg("../../potentially_germany_grad.json", "../../boundary_ids.pgm");
    const std::vector<double> weights{6.22485, 9.99625,   9.99997, 0.0000887, 4.78272, 0.000177, 7.80954,
                                      7.84763, 0.0000798, 9.99982, 9.99987,   3.77446, 10,       2.92137};
    const std::vector<double> sigmas{0.00114814, 0.0000000504, 15, 15, 14.9999, 0.000246785, 14.4189, 14.9999};
    wg.apply_weights(weights);

    std::cerr << "Setup: Read potential.\n" << std::flush;
    {
        const auto fname = "../../potentially_germany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            return 1;
        }
        else {
            potential = 8 * mio::mpm::read_pgm(ifile);
            ifile.close();
        }
    }
    std::cerr << "Setup: Read metaregions.\n" << std::flush;
    {
        const auto fname = "../../metagermany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            return 1;
        }
        else {
            metaregions = mio::mpm::read_pgm_raw(ifile).first;
            ifile.close();
        }
    }
    return 0;
}