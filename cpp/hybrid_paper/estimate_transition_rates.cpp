#include "hybrid_paper/weighted_gradient.h"
#include "memilio/io/cli.h"
#include "mpm/abm.h"
#include "mpm/model.h"
#include "models/mpm/potentials/map_reader.h"
#include "mpm/potentials/potential_germany.h"
#include "hybrid_paper/weighted_potential.h"
#include "hybrid_paper/initialization.h"

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

void print_transition_rates(std::vector<TransitionRate<InfectionState>>& transition_rates)
{
    std::cout << "\n Transition Rates: \n";
    std::cout << "status from to factor \n";
    for (auto& rate : transition_rates) {
        std::cout << "S"
                  << "  " << rate.from << "  " << rate.to << "  " << rate.factor << "\n";
    }
}

template <class Potential>
void calculate_transition_rates(ABM<Potential>& abm, size_t num_runs, double tmax, size_t num_regions)
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

        //add calculated transition rates
        for (auto& tr_rate : estimated_transition_rates[run]) {
            tr_rate.factor = sim.get_model().number_transitions(tr_rate) / (abm.populations.size() * tmax);
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

    print_transition_rates(mean_transition_rates);
}

} // namespace paper
} // namespace mpm
} // namespace mio
int main(int argc, char** argv)
{
    auto cli_result = mio::command_line_interface<Tmax>(argv[0], argc, argv);
    if (!cli_result) {
        std::cerr << cli_result.error().formatted_message() << "\n";
        return cli_result.error().code().value();
    }
    auto tmax = cli_result.value().get<Tmax>();

    Eigen::MatrixXi metaregions;

    WeightedGradient wg("../../potentially_germany_grad.json", "../../boundary_ids.pgm");

    const std::vector<double> weights{1000, 800,  850.076, 717.145, 1000, 50,      611.022,
                                      160,  1000, 500,     100,     1200, 725.818, 590.303};
    const std::vector<double> sigmas{15, 29, 15, 15, 28, 35, 43, 30};
    const double slope = 2.0;
    wg.apply_weights(weights);

    const Eigen::Vector2d centre = {wg.gradient.rows() / 2.0, wg.gradient.cols() / 2.0}; // centre of the wg.gradient
    for (Eigen::Index i = 0; i < wg.gradient.rows(); i++) {
        for (Eigen::Index j = 0; j < wg.gradient.cols(); j++) {
            if (wg.base_gradient(i, j) == Eigen::Vector2d{0, 0}) {
                auto direction    = (Eigen::Vector2d{i, j} - centre).normalized();
                wg.gradient(i, j) = slope * direction;
            }
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

    // std::vector<mio::mpm::ABM<GradientGermany<mio::mpm::paper::InfectionState>>::Agent> agents;
    // for (Eigen::Index i = 0; i < metaregions.rows(); i += 2) {
    //     for (Eigen::Index j = 0; j < metaregions.cols(); j += 2) {
    //         if (metaregions(i, j) != 0) {
    //             agents.push_back({{i, j}, mio::mpm::paper::InfectionState::S, metaregions(i, j) - 1});
    //         }
    //     }
    // }

    std::string init_file = "/group/HPC/Gruppen/PSS/Modelle/Hybrid Models/Papers, Theses, "
                            "Posters/2023_Paper_Spatio-temporal_hybrid/initializations/initSusceptible9375.json";

    std::vector<mio::mpm::ABM<GradientGermany<mio::mpm::paper::InfectionState>>::Agent> agents;
    read_initialization(init_file, agents);

    mio::mpm::ABM<GradientGermany<mio::mpm::paper::InfectionState>> model(agents, {}, wg.gradient, metaregions,
                                                                          {mio::mpm::paper::InfectionState::D}, sigmas);

    calculate_transition_rates(model, 10, tmax, metaregions.maxCoeff());

    return 0;
}
