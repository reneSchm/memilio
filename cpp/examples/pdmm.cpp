#include "memilio/compartments/simulation.h"
#include "memilio/epidemiology/populations.h"
#include "memilio/compartments/compartmentalmodel.h"
#include "memilio/math/eigen.h"
#include "memilio/utils/logging.h"
#include "memilio/utils/parameter_distributions.h"
#include "memilio/utils/parameter_set.h"
#include "memilio/utils/random_number_generator.h"
#include "memilio/utils/time_series.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

enum Order
{
    first,
    second
};

enum Status
{
    S,
    I,
    R,
    Count
};

struct AdoptionRate {
    Status from; // i
    Status to; // j
    double factor; // gammahat_{ij}^k
    Order order; // first or second

    template <class Population>
    constexpr double operator()(const Population& p) const
    {
        if (order == Order::second) {
            return factor * p.array()[from] * p.array()[to] / p.array().sum();
        }
        else {
            return factor * p.array()[from];
        }
    }
};

struct AdoptionRates {
    // consider std::map<<i, j>, rate>, if access by index is ever needed
    using Type = std::vector<AdoptionRate>;
    const static std::string name()
    {
        return "AdoptionRates";
    }
};

struct TransitionRate {
    Status status; // i
    int from; // k
    int to; // l
    double factor; // lambda_i^{kl}

    template <class Population>
    constexpr double operator()(const Population& p) const
    {
        return factor * p.array()[status];
    }
};

struct TransitionRates {
    // consider a more dense format, e.g. a matrix of size n_sudomains × Status::Count
    using Type = std::vector<TransitionRate>;
    const static std::string name()
    {
        return "TransitionRates";
    }
};

using Params = mio::ParameterSet<AdoptionRates>;

class PDMM : public mio::CompartmentalModel<Status, mio::Populations<Status>, Params>
{
    using Base = mio::CompartmentalModel<Status, mio::Populations<Status>, Params>;

public:
    PDMM()
        : Base(Populations({Status::Count}, 0.), ParameterSet())
    {
    }
    void get_derivatives(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd>, double t,
                         Eigen::Ref<Eigen::VectorXd> dxdt) const override
    {
        auto& params = this->parameters;
        for (auto& rate : params.template get<AdoptionRates>()) {
            dxdt[rate.from] -= rate(x);
            dxdt[rate.to] += rate(x);
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
        m_simulations.reserve(models.size());
        for (auto&& m : models) {
            m_simulations.push_back(Simulation<PDMM>(m, t0, dt));
        }
    }
    std::vector<Eigen::Ref<Eigen::VectorXd>> advance(double tmax, double dt_max = std::numeric_limits<double>::max())
    {
        // determine how long we wait until the next transition occurs
        double waiting_time = draw_waiting_time();
        double remaining_time;
        // iterate time by increments of m_dt
        while (m_t0 < tmax) {
            m_dt           = std::min({m_dt, tmax - m_t0});
            remaining_time = m_dt; // xi * Delta-t
            // check if one or more transitions occur during this time step
            if (waiting_time < m_dt) { // (at least one) event occurs
                std::vector<double> rates(transition_rates.size()); // lambda_m (for all m)
                // perform transition(s)
                do {
                    //double event_time = m_t0 + (m_dt - remaining_time) + waiting_time;
                    m_t0 += waiting_time; // event time t**
                    // advance all locations
                    for (auto&& sim : m_simulations) {
                        sim.advance(m_t0);
                    }
                    // compute current transition rates
                    std::transform(transition_rates.begin(), transition_rates.end(), rates.begin(), [&](auto&& r) {
                        // we should normalize here by dividing by the cumulative transition rate,
                        // but the DiscreteDistribution below effectively does that for us
                        return compute_rate(r);
                    });
                    // draw transition event and execute it
                    int event = mio::DiscreteDistribution<int>::get_instance()(rates);
                    perform_transition(event);
                    // draw new waiting time
                    remaining_time -= waiting_time;
                    /* mio::log_info("time {} \t event: {}, {}->{}", event_time, transition_rates[event].status,
                                  transition_rates[event].from, transition_rates[event].to); */
                    waiting_time = draw_waiting_time();
                    // repeat, if another event occurs in the remaining time interval
                } while (waiting_time < remaining_time);
            }
            else { // no event occurs
                // reduce waiting time for the next step
                //normalized_waiting_time -= cumulative_transition_rate * m_dt;
                waiting_time -= m_dt;
            }
            // advance all locations
            for (auto&& sim : m_simulations) {
                sim.advance(m_t0 + remaining_time);
            }
            m_t0 += remaining_time;
            update_dt(dt_max);
        }
        // gather and return simulation results
        std::vector<Eigen::Ref<Eigen::VectorXd>> results;
        for (size_t i = 0; i < m_simulations.size(); i++) {
            results.emplace_back(m_simulations[i].get_result().get_last_value());
        }
        return results;
    }

    mio::TimeSeries<double>& get_result(size_t location)
    {
        return m_simulations[location].get_result();
    }

    TransitionRates::Type transition_rates;

private:
    void perform_transition(int event)
    {
        // get the rate corresponding to the event
        const auto& r = transition_rates[event];
        // transitioning one person at the current time
        auto& value = m_simulations[r.from].get_result().get_last_value()[static_cast<Eigen::Index>(r.status)];
        value -= 1;
        if (value < 0) {
            mio::log_error("Transition from {} to {} with status {} caused negative value.", r.from, r.to, r.status);
        }
        m_simulations[r.to].get_result().get_last_value()[static_cast<Eigen::Index>(r.status)] += 1;
    }
    double compute_rate(const TransitionRate& r) const
    {
        return r(m_simulations[r.from].get_result().get_last_value());
    }
    double draw_waiting_time()
    {
        // compute the current cumulative transition rate
        double ctr = 0; // Lambda
        for (auto rate : transition_rates) {
            ctr += compute_rate(rate);
        }
        // draw the normalized waiting time
        double nwt = mio::ExponentialDistribution<double>::get_instance()(1.0); // tau'
        return nwt / ctr;
    }
    void update_dt(double dt_max)
    {
        m_dt = dt_max;
        for (auto&& sim : m_simulations) {
            if (sim.get_dt() < m_dt) {
                m_dt = sim.get_dt();
            }
        }
    }
    std::vector<mio::Simulation<PDMM>> m_simulations;
    double m_t0, m_dt;
};

template <class SC>
void print_to_terminal(const mio::TimeSeries<SC>& results, const std::vector<std::string>& state_names)
{
    printf("%-16s   ", "Time");
    for (size_t k = 0; k < state_names.size(); k++) {
        printf(" %-16s ", state_names[k].data()); // print underlying char*
    }
    auto num_points = static_cast<size_t>(results.get_num_time_points());
    for (size_t i = 0; i < num_points; i++) {
        printf("\n%16.6f ", results.get_time(i));
        auto res_i = results.get_value(i);
        for (size_t j = 0; j < state_names.size(); j++) {
            printf(" %16.6f ", res_i[j]);
        }
    }
    printf("\n");
}

int main()
{
    /*** CONFIG ***/
    const int n_subdomains = 2;
    mio::set_log_level(mio::LogLevel::warn);
    /*** END CONFIG ***/

    auto pop_size = PDMM::Compartments::Count;

    // vector of locations k; list entries = {staus_from, status_to, gammahat, order}
    std::vector<std::list<AdoptionRate>> adoption_rates(n_subdomains);
    // vector {status, location_from, location_to, lambda}
    TransitionRates::Type transition_rates;

    /*** CONFIG ***/
    adoption_rates[0].push_back({Status::S, Status::I, 0.3, Order::second});
    adoption_rates[0].push_back({Status::I, Status::R, 0.1, Order::first});
    adoption_rates[1].push_back({Status::S, Status::I, 1, Order::second});
    adoption_rates[1].push_back({Status::I, Status::R, 0.08, Order::first});

    double kappa     = 0.001;
    transition_rates = {{Status::S, 0, 1, 0.5 * kappa},  {Status::I, 0, 1, 0.1 * kappa},
                        {Status::R, 0, 1, 0.1 * kappa},  {Status::S, 1, 0, 0.1 * kappa},
                        {Status::I, 1, 0, 0.02 * kappa}, {Status::R, 1, 0, 0.2 * kappa}};

    std::vector<std::vector<double>> populations{{1900, 100, 0}, {2000, 0, 0}};
    /*** END CONFIG ***/

    std::vector<PDMM> local_models(n_subdomains);

    for (int k = 0; k < n_subdomains; k++) {
        local_models[k].parameters.get<AdoptionRates>().reserve(adoption_rates[k].size());
        for (auto& r : adoption_rates[k]) {
            local_models[k].parameters.get<AdoptionRates>().emplace_back(r);
        }
        for (int i = 0; i < pop_size; i++) {
            local_models[k].populations.array()[i] = populations[k][i];
        }
    }

    int i = 0;
    /* for (auto& m : local_models) {
        auto results = mio::simulate(0, 10, 0.1, m);
        std::cout << "Model " << ++i << "/" << n_subdomains << "\n";
        print_to_terminal(results, {"S", "I", "R"});
    } */

    auto sim             = mio::Simulation<std::vector<PDMM>>(local_models, 0, 0.1);
    sim.transition_rates = transition_rates;
    sim.advance(100);

    for (i = 0; i < n_subdomains; i++) {
        std::cout << "Global Model " << i + 1 << "/" << n_subdomains << "\n";
        print_to_terminal(sim.get_result(i), {"S", "I", "R"});
    }

    return 0;
}