#include "hybrid_paper/library/weighted_gradient.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/map_reader.h"
#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/potentials/potential_germany.h"
#include "mpm/abm.h"
#include "mpm/model.h"

#include "memilio/io/cli.h"

#include <numeric>
#include <vector>

struct Tmax {
    using Type = double;
    const static std::string name()
    {
        return "Tmax";
    }
    const static std::string alias()
    {
        return "tmax";
    }
};

namespace mio
{
namespace mpm
{
namespace paper
{

std::vector<TransitionRate<InfectionState>> add_transition_rates(std::vector<TransitionRate<InfectionState>>& v1,
                                                                 std::vector<TransitionRate<InfectionState>>& v2)
{
    std::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(),
                   [](TransitionRate<InfectionState>& t1, TransitionRate<InfectionState>& t2) {
                       return TransitionRate<InfectionState>{t1.status, t1.from, t1.to, t1.factor + t2.factor};
                   });
    return v1;
}

void print_transition_rates(std::vector<TransitionRate<InfectionState>>& transition_rates, bool print_to_file)
{
    if (print_to_file) {

        std::string fname = mio::base_dir() + "transition_rates.txt";
        std::ofstream ofile(fname);

        ofile << "from to factor \n";
        for (auto& rate : transition_rates) {
            ofile << rate.from << "  " << rate.to << "  " << rate.factor << "\n";
        }
    }
    else {
        std::cout << "\n Transition Rates: \n";
        std::cout << "status from to factor \n";
        for (auto& rate : transition_rates) {
            std::cout << "S"
                      << "  " << rate.from << "  " << rate.to << "  " << rate.factor << "\n";
        }
    }
}

template <class Potential>
std::vector<TransitionRate<InfectionState>> calculate_transition_rates(ABM<Potential>& abm, size_t num_runs,
                                                                       double tmax, size_t num_regions,
                                                                       std::vector<double>& ref_pops)
{
    std::vector<std::vector<TransitionRate<InfectionState>>> estimated_transition_rates(num_runs);

    std::vector<TransitionRate<InfectionState>> zero_transition_rates;

    for (int i = 0; i < num_regions; ++i) {
        for (int j = 0; j < num_regions; ++j) {
            if (i != j) {
                zero_transition_rates.push_back({InfectionState::S, mio::mpm::Region(i), mio::mpm::Region(j), 0});
            }
        }
    }
    std::fill_n(estimated_transition_rates.begin(), num_runs, zero_transition_rates);
    for (size_t run = 0; run < num_runs; ++run) {
        std::cerr << "run number: " << run << "\n" << std::flush;
        mio::Simulation<ABM<Potential>> sim(abm, 0, 0.05);
        sim.advance(tmax);

        double scaling_factor = std::accumulate(ref_pops.begin(), ref_pops.end(), 0.0) / abm.populations.size();
        //add calculated transition rates
        for (auto& tr_rate : estimated_transition_rates[run]) {
            auto region_pop = ref_pops[tr_rate.from.get()] / scaling_factor;
            tr_rate.factor  = sim.get_model().number_transitions(tr_rate) / (region_pop * tmax);
        }
    }

    auto mean_transition_rates = std::accumulate(estimated_transition_rates.begin(), estimated_transition_rates.end(),
                                                 zero_transition_rates, add_transition_rates);

    double denominator{(1 / (double)num_runs)};
    std::transform(
        mean_transition_rates.begin(), mean_transition_rates.end(), mean_transition_rates.begin(),
        [&denominator](mio::mpm::TransitionRate<InfectionState>& rate) {
            return mio::mpm::TransitionRate<InfectionState>{rate.status, rate.from, rate.to, denominator * rate.factor};
        });

    print_transition_rates(mean_transition_rates, true);
    return mean_transition_rates;
}

} // namespace paper
} // namespace mpm
} // namespace mio

template <class Model>
void print_movement(std::vector<typename Model::Agent> agents, Model& model, const double tmax, double dt)
{
    double t = 0;
    auto sim = mio::Simulation<Model>(model, t, dt);
    while (t < tmax) {
        for (auto& agent : sim.get_model().populations) {
            std::cout << agent.position[0] << " " << agent.position[1] << " ";
            sim.get_model().move(t, dt, agent);
        }
        std::cout << "\n";
        t += dt;
    }
}

std::string colorize(double a, double b)
{
    std::stringstream ss("");
    if (a / b < 1) {
        double proc = 1 - a / b;
        if (proc <= 0.05) {
            ss << "\033[32m"; // green
        }
        else if (proc <= 0.15) {
            ss << "\033[33m"; // yellow
        }
        else {
            ss << "\033[31m"; // red
        }
    }
    else {
        double proc = 1 - b / a;
        if (proc <= 0.05) {
            ss << "\033[42m"; // green
        }
        else if (proc <= 0.15) {
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
    using Model  = mio::mpm::ABM<CommutingPotential<StochastiK, mio::mpm::paper::InfectionState>>;
    using Status = mio::mpm::paper::InfectionState;
    // auto cli_result = mio::command_line_interface<Tmax>(argv[0], argc, argv);
    // if (!cli_result) {
    //     std::cerr << cli_result.error().formatted_message() << "\n";
    //     return cli_result.error().code().value();
    // }
    auto tmax = 100; //cli_result.value().get<Tmax>();

    Eigen::MatrixXi metaregions;

    WeightedGradient wg(mio::base_dir() + "potentially_germany_grad.json", mio::base_dir() + "boundary_ids.pgm");

    const std::vector<int> county_ids   = {233, 228, 242, 223, 238, 232, 231, 229};
    Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();
    Eigen::MatrixXd reference_transitions_daily =
        get_transition_matrix_daily_total(mio::base_dir() + "data/mobility/").value();
    auto ref_pops = std::vector<double>{218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    auto ref_pop  = std::accumulate(ref_pops.begin(), ref_pops.end(), 0.0);

    int num_regions = county_ids.size();
    int num_weights = 14;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(num_regions, num_regions);
    commute_weights.setZero();
    // for (int i = 0; i < num_regions; i++) {
    //     commute_weights(i, i) = 1;
    // }
    for (int i = 0; i < num_regions; i++) {
        for (int j = 0; j < num_regions; j++) {
            if (i != j) {
                commute_weights(i, j) = reference_commuters(
                    county_ids[i],
                    county_ids[j]); //commute_weights(i,j)/persons_per_agents + (commute_weights(j,i)/ref_pop)
            }
        }
        commute_weights(i, i) = ref_pops[i] - commute_weights.row(i).sum();
    }

    std::cerr << "Setup: Read metaregions.\n" << std::flush;
    mio::log_thread_local_rng_seeds(mio::LogLevel::info);
    {
        const auto fname = mio::base_dir() + "metagermany.pgm";
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

    auto sigmas  = std::vector<double>(num_regions, 10);
    auto weights = std::vector<double>(14, 500);

    wg.apply_weights(weights);

    StochastiK k_provider(commute_weights, metaregions, {metaregions});
    std::cerr << k_provider.metaregion_commute_weights << "\n";

    std::string init_file = "/group/HPC/Gruppen/PSS/Modelle/Hybrid Models/Papers, Theses, "
                            "Posters/2023_Paper_Spatio-temporal_hybrid/initializations/initSusceptible9375.json";

    std::vector<Model::Agent> agents;

    // std::vector<Eigen::Vector2d> positions{{592, 648}, {840, 604},  {896, 700}, {426, 136},  {626, 568},
    //                                        {312, 688}, {880, 1166}, {682, 776}, {1088, 864}, {950, 406}};
    // for (auto p : positions) {
    //     agents.push_back({p, Status::S, int(metaregions(p[0], p[1]) - 1)});
    // }

    read_initialization(init_file, agents);
    size_t num_agents        = agents.size();
    double persons_per_agent = ref_pop / num_agents;

    std::cout << "persons per agent: " << persons_per_agent << "\n";

    Model model(k_provider, agents, {}, wg.gradient, metaregions, {Status::D}, sigmas);

    //print_movement<Model>(agents, model, tmax, 0.1);

    auto rates = calculate_transition_rates(model, 10, tmax, metaregions.maxCoeff(), ref_pops);

    //check
    for (auto rate : rates) {
        //rates
        std::cout << rate.from << "->" << rate.to << ": rel: "
                  << colorize(rate.factor,
                              (commute_weights(static_cast<size_t>(rate.from), static_cast<size_t>(rate.to)) +
                               commute_weights(static_cast<size_t>(rate.to), static_cast<size_t>(rate.from))) /
                                  ref_pops[rate.from.get()])
                  << std::endl;
    }

    return 0;
}
