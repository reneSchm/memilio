#include "mpm/pdmm.h"
#include "mpm/utility.h"

enum class InfectionState
{
    S,
    I,
    R,
    Count
};

int main()
{
    const unsigned regions = 2;

    using namespace mio::mpm;
    using Model  = PDMModel<regions, InfectionState>;
    using Status = Model::Compartments;

    Model model;

    /*** CONFIG ***/
    mio::set_log_level(mio::LogLevel::info);
    ScalarType kappa                                 = 0.001;
    std::vector<std::vector<ScalarType>> populations = {{950, 50, 0}, {1000, 0, 0}};
    model.parameters.get<AdoptionRates<Status>>()    = {{Status::S, Status::I, 0.3, Order::second, 0},
                                                        {Status::I, Status::R, 0.1, Order::first, 0},
                                                        {Status::S, Status::I, 1, Order::second, 1},
                                                        {Status::I, Status::R, 0.08, Order::first, 1}};
    model.parameters.get<TransitionRates<Status>>()  = {{Status::S, 0, 1, 0.5 * kappa},  {Status::I, 0, 1, 0.1 * kappa},
                                                        {Status::R, 0, 1, 0.1 * kappa},  {Status::S, 1, 0, 0.1 * kappa},
                                                        {Status::I, 1, 0, 0.02 * kappa}, {Status::R, 1, 0, 0.2 * kappa}};
    /*** CONFIG END ***/

    // assign population
    for (size_t k = 0; k < regions; k++) {
        for (int i = 0; i < static_cast<size_t>(Status::Count); i++) {
            model.populations[{static_cast<Region>(k), static_cast<Status>(i)}] = populations[k][i];
        }
    }

    auto result = mio::simulate(0, 100, 0.1, model);

    print_to_terminal(result, {"S", "I", "R", "S", "I", "R"});

    return 0;
}