#include "transition_rate_estimation.h"

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

} // namespace paper

} // namespace mpm

} // namespace mio