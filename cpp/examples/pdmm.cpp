#include "memilio/compartments/simulation.h"
#include "memilio/epidemiology/populations.h"
#include "memilio/compartments/compartmentalmodel.h"
#include "memilio/math/eigen.h"
#include "memilio/utils/parameter_distributions.h"
#include "memilio/utils/parameter_set.h"
#include "memilio/utils/random_number_generator.h"
#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

template <class SC>
void print_to_terminal(const mio::TimeSeries<SC>& results, const std::vector<std::string>& state_names)
{
    printf("| %-16s |", "Time");
    for (size_t k = 0; k < state_names.size(); k++) {
        printf(" %-16s |", state_names[k].data()); // print underlying char*
    }
    auto num_points = static_cast<size_t>(results.get_num_time_points());
    for (size_t i = 0; i < num_points; i++) {
        printf("\n| %16.6f |", results.get_time(i));
        auto res_i = results.get_value(i);
        for (size_t j = 0; j < state_names.size(); j++) {
            printf(" %16.6f |", res_i[j]);
        }
    }
    printf("\n");
}

struct AdoptionRate {
    double factor;
    bool second_order;
    int i;
    int j;

    template <class Population>
    constexpr double operator()(const Population& p) const
    {
        if (second_order) {
            return factor * p.array()[i] * p.array()[j];
        }
        else {
            return factor * p.array()[i];
        }
    }
};

struct AdoptionRates {
    using Type = std::vector<std::vector<AdoptionRate>>;
    const static std::string name()
    {
        return "AdoptionRates";
    }
};

enum Status
{
    S,
    I,
    R,
    Count
};

class PDMM : public mio::CompartmentalModel<Status, mio::Populations<Status>, mio::ParameterSet<AdoptionRates>>
{
    using Base = mio::CompartmentalModel<Status, mio::Populations<Status>, mio::ParameterSet<AdoptionRates>>;

public:
    PDMM()
        : Base(Populations({Status::Count}, 0.), ParameterSet())
    {
    }
    void get_derivatives(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd>, double t,
                         Eigen::Ref<Eigen::VectorXd> dxdt) const override
    {
        auto& params = this->parameters;

        for (Eigen::Index i = 0; i < x.size(); i++) {
            for (Eigen::Index j = 0; j < x.size(); j++) {
                // TODO: replace AdoptionRates by sparse pattern
                dxdt[i] -= params.template get<AdoptionRates>()[i][j](x);
                dxdt[j] += params.template get<AdoptionRates>()[i][j](x);
            }
        }
    }
};

template <>
class mio::Simulation<std::vector<PDMM>>
{
public:
    Simulation(const std::vector<PDMM>& models, double t0 = 0., double dt = 0.1)
        : m_simulations()
        , m_t0(t0)
        , m_dt(dt)
    {
        std::vector<Simulation<PDMM>> sims;
        sims.reserve(models.size());
        for (auto&& m : models) {
            sims.push_back(Simulation<PDMM>(m, t0, dt));
        }
    }
    /* Eigen::Ref<Eigen::VectorXd> advance(double tmax)
    {
        double next_jump_time = mio::ExponentialDistribution<double>::get_instance()(1); // tau'
        double cumulative_transition_rate; // TODO: Lambda
        while (cumulative_transition_rate * m_dt <= next_jump_time) {
            next_jump_time -= cumulative_transition_rate * m_dt;

            for (auto&& sim : *m_simulations) {
                sim.advance(m_t0 + m_dt);
            }
            // TODO: update m_dt?
            m_t0 += m_dt;
        }
        int event = mio::DiscreteDistribution<int>::get_instance()(rates / cumulative_transition_rate)
            do_jump(event, t = t_n + tau_dash / Lambda)

                while (another event in this time step)
        {
            do_jump
        }
    } */

private:
    std::unique_ptr<std::vector<mio::Simulation<PDMM>>> m_simulations;
    double m_t0, m_dt;
};

int main()
{
    /*** CONFIG ***/
    const int n_subdomains = 2;
    /*** END CONFIG ***/

    auto pop_size = PDMM::Compartments::Count;

    // index: k=loc, i=staus_from, j=status_to; value: {factor, order} of adoption i->j in k
    std::vector<std::vector<std::vector<std::pair<double, int>>>> gammas{};
    gammas.resize(n_subdomains);
    for (auto&& locs : gammas) {
        locs.resize(pop_size, std::vector<std::pair<double, int>>(pop_size, {0, 0}));
    }

    /*** CONFIG ***/
    gammas[0][Status::S][Status::I] = {0.5, 2};
    gammas[0][Status::I][Status::R] = {0.001, 1};
    gammas[1][Status::S][Status::I] = {0.001, 2};
    gammas[1][Status::I][Status::R] = {0.5, 1};
    std::vector<double> pops{1900, 100, 0};
    /*** END CONFIG ***/

    std::vector<PDMM> local_models(n_subdomains);

    for (int k = 0; k < n_subdomains; k++) {
        local_models[k].parameters.get<AdoptionRates>() =
            AdoptionRates::Type(pop_size, std::vector<AdoptionRate>(pop_size));
        for (int i = 0; i < pop_size; i++) {
            for (int j = 0; j < pop_size; j++) {
                local_models[k].parameters.get<AdoptionRates>()[i][j] =
                    AdoptionRate{gammas[k][i][j].first / std::accumulate(pops.begin(), pops.end(), 0),
                                 (gammas[k][i][j].second == 2), i, j};
            }
        }
    }
    for (auto& m : local_models) {
        for (int i = 0; i < pop_size; i++) {
            m.populations.array()[i] = pops[i];
        }
    }

    int i = 0;
    for (auto& m : local_models) {
        auto results = mio::simulate(0, 10, 0.1, m);
        std::cout << "Model " << ++i << "/" << n_subdomains << "\n";
        print_to_terminal(results, {"S", "I", "R"});
    }

    return 0;
}