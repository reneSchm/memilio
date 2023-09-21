#include "mpm/abm.h"
#include "hybrid_paper/infection_state.h"
#include "mpm/model.h"
#include "models/mpm/potentials/map_reader.h"
#include "mpm/potentials/potential_germany.h"

namespace mio {
namespace mpm{
namespace paper{

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
        std::cout << "S" << "  " << rate.from << "  " << rate.to << "  " << rate.factor << "\n";
    }
}

template <class Potential>
void calculate_transition_rates(ABM<Potential>& abm, size_t num_runs, double tmax, size_t num_regions){
        std::vector<std::vector<TransitionRate<InfectionState>>> estimated_transition_rates(num_runs);

        std::vector<TransitionRate<InfectionState>> zero_transition_rates;

        for(int i=0;i<num_regions; ++i){
            for(int j=0; j<num_regions;++j){
                if (i != j) {
                    zero_transition_rates.push_back({InfectionState::S, mio::mpm::Region(i), mio::mpm::Region(j), 0});
                }
            }
        }
        std::fill_n(estimated_transition_rates.begin(), num_runs, zero_transition_rates);
        for(size_t run = 0; run<num_runs; ++run){
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

}
}
}
int main()
{
    Eigen::MatrixXd potential;
    Eigen::MatrixXi metaregions;

    std::cerr << "Setup: Read potential.\n" << std::flush;
    {
        const auto fname =
            "../../potentially_germany.pgm";
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

    std::vector<mio::mpm::ABM<PotentialGermany<mio::mpm::paper::InfectionState>>::Agent> agents;
    for (Eigen::Index i = 0; i < metaregions.rows(); i += 2) {
        for (Eigen::Index j = 0; j < metaregions.cols(); j += 2) {
                if (metaregions(i, j) != 0) {
                    agents.push_back({{i, j}, mio::mpm::paper::InfectionState::S, metaregions(i, j) - 1});
                }
            }
        }

    mio::mpm::ABM<PotentialGermany<mio::mpm::paper::InfectionState>> model(agents, {}, potential, metaregions);

    calculate_transition_rates(model, adoption_rates, 10, 100, 8);

    return 0;
}
