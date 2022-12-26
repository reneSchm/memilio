#include "memilio/compartments/simulation.h"
#include "memilio/config.h"
#include "memilio/epidemiology/populations.h"
#include "memilio/compartments/compartmentalmodel.h"
#include "memilio/math/eigen.h"
#include "memilio/utils/logging.h"
#include "memilio/utils/parameter_distributions.h"
#include "memilio/utils/parameter_set.h"
#include "memilio/utils/random_number_generator.h"
#include "memilio/utils/time_series.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

enum Order
{
    first,
    second
};

enum InfectionState
{
    S,
    I,
    R,
    Count
};

template <class Status>
struct AdoptionRate {
    Status from; // i
    Status to; // j
    ScalarType factor; // gammahat_{ij}^k
    Order order; // first or second
    size_t location; // k

    template <class Population>
    constexpr ScalarType operator()(const Population& p) const
    {
        if (order == Order::second) {
            return factor * p.array()[from] * p.array()[to] / p.array().sum();
        }
        else {
            return factor * p.array()[from];
        }
    }
};

template <class Status>
struct AdoptionRates {
    // consider std::map<<i, j>, rate>, if access by index is ever needed
    using Type = std::vector<AdoptionRate<Status>>;
    const static std::string name()
    {
        return "AdoptionRates";
    }
};

template <class Status>
struct TransitionRate {
    Status status; // i
    size_t from; // k
    size_t to; // l
    ScalarType factor; // lambda_i^{kl}

    template <class Population>
    constexpr ScalarType operator()(const Population& p) const
    {
        return factor * p.array()[status];
    }
};

template <class Status>
struct TransitionRates {
    // consider a more dense format, e.g. a matrix of size n_sudomains × Status::Count
    using Type = std::vector<TransitionRate<Status>>;
    const static std::string name()
    {
        return "TransitionRates";
    }
};

template <class Status>
class PDMNode
    : public mio::CompartmentalModel<Status, mio::Populations<Status>, mio::ParameterSet<AdoptionRates<Status>>>
{
    using Base = mio::CompartmentalModel<Status, mio::Populations<Status>, mio::ParameterSet<AdoptionRates<Status>>>;

public:
    PDMNode()
        : Base(typename Base::Populations({Status::Count}, 0.), typename Base::ParameterSet())
    {
    }
    void get_derivatives(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd>, ScalarType t,
                         Eigen::Ref<Eigen::VectorXd> dxdt) const override
    {
        const auto& params = this->parameters;
        for (const auto& rate : params.template get<AdoptionRates<Status>>()) {
            dxdt[rate.from] -= rate(x);
            dxdt[rate.to] += rate(x);
        }
    }
};

template <class Status>
class PDMModel
{
public:
    PDMModel(const std::vector<std::array<ScalarType, Status::Count>>& populations,
             const std::list<AdoptionRate<Status>>& adoption_rates,
             const std::list<TransitionRate<Status>>& transition_rates)
        : m_transition_rates(transition_rates.begin(), transition_rates.end())
        , m_metapopulations(populations.size())
    {
        assert(max_location_index(adoption_rates, transition_rates) < populations.size());
        for (const auto& adoption_rate : adoption_rates) {
            m_metapopulations[adoption_rate.location].parameters.template get<AdoptionRates<Status>>().push_back(
                adoption_rate);
        }
        for (size_t k = 0; k < populations.size(); k++) {
            for (size_t i = 0; i < Status::Count; i++) {
                m_metapopulations[k].populations.array()[i] = populations[k][i];
            }
        }
    }

    const std::vector<PDMNode<Status>>& get_metapopulations() const
    {
        return m_metapopulations;
    }

    const typename TransitionRates<Status>::Type& get_transition_rates() const
    {
        return m_transition_rates;
    }

    // TODO: (de-)serialize

private:
    static constexpr size_t max_location_index(const std::list<AdoptionRate<Status>>& ars,
                                               const std::list<TransitionRate<Status>>& trs)
    {
        size_t max = 0;
        for (auto& ar : ars) {
            max = std::max(max, ar.location);
        }
        for (auto& tr : trs) {
            max = std::max({max, tr.from, tr.to});
        }
        return max;
    }

    const typename TransitionRates<Status>::Type m_transition_rates;
    std::vector<PDMNode<Status>> m_metapopulations;
};

template <class Status>
class Simulation
{
public:
    Simulation(const PDMModel<Status>& model, ScalarType t0 = 0., ScalarType dt = 0.1)
        : m_dt(dt)
        , m_t0(t0)
        , m_model(model)
        , m_result_iterator(Status::Count, 0)
        , m_results(model.get_metapopulations().size() * Status::Count)
        , m_simulations()
    {
        m_simulations.reserve(model.get_metapopulations().size());
        for (const auto& m : model.get_metapopulations()) {
            m_simulations.push_back(mio::Simulation<PDMNode<Status>>(m, t0, dt));
        }
        m_results.add_time_point(t0);
        for (size_t i = 0; i < m_simulations.size(); i++) {
            m_results.get_last_value().segment(i * Status::Count, Status::Count) =
                m_simulations[i].get_result().get_last_value();
        }
    }
    Eigen::Ref<Eigen::VectorXd> advance(ScalarType tmax, ScalarType dt_max = std::numeric_limits<ScalarType>::max())
    {
        // determine how long we wait until the next transition occurs
        ScalarType waiting_time = draw_waiting_time();
        ScalarType remaining_time;
        // iterate time by increments of m_dt
        while (m_t0 < tmax) {
            m_dt           = std::min({m_dt, tmax - m_t0});
            remaining_time = m_dt; // xi * Delta-t
            // check if one or more transitions occur during this time step
            if (waiting_time < m_dt) { // (at least one) event occurs
                std::vector<ScalarType> rates(m_model.get_transition_rates().size()); // lambda_m (for all m)
                // perform transition(s)
                do {
                    m_t0 += waiting_time; // event time t**
                    // advance all locations
                    for (auto&& sim : m_simulations) {
                        sim.advance(m_t0);
                    }
                    // compute current transition rates
                    std::transform(m_model.get_transition_rates().begin(), m_model.get_transition_rates().end(),
                                   rates.begin(), [&](auto&& r) {
                                       // we should normalize here by dividing by the cumulative transition rate,
                                       // but the DiscreteDistribution below effectively does that for us
                                       return compute_rate(r);
                                   });
                    // draw transition event and execute it
                    int event = mio::DiscreteDistribution<int>::get_instance()(rates);
                    perform_transition(event);
                    update_results();
                    // draw new waiting time
                    remaining_time -= waiting_time;
                    waiting_time = draw_waiting_time();
                    // repeat, if another event occurs in the remaining time interval
                } while (waiting_time < remaining_time);
            }
            else { // no event occurs
                // reduce waiting time for the next step
                waiting_time -= m_dt;
            }
            // advance time
            m_t0 += remaining_time;
            // advance all locations
            for (auto&& sim : m_simulations) {
                sim.advance(m_t0);
            }
            update_results();
            update_dt(dt_max);
        }
        return m_results.get_last_value();
    }

    mio::TimeSeries<ScalarType>& get_result()
    {
        return m_results;
    }

    const mio::TimeSeries<ScalarType>& get_result() const
    {
        return m_results;
    }

    mio::TimeSeries<ScalarType>& get_result(size_t location)
    {
        return m_simulations[location].get_result();
    }

    const mio::TimeSeries<ScalarType>& get_result(size_t location) const
    {
        return m_simulations[location].get_result();
    }

private:
    /// @brief perform the given transition event, moving a single "agent"
    void perform_transition(int event)
    {
        // get the rate corresponding to the event
        const auto& r = m_model.get_transition_rates()[event];
        // transitioning one person at the current time
        auto& value = m_simulations[r.from].get_result().get_last_value()[static_cast<Eigen::Index>(r.status)];
        value -= 1;
        if (value < 0) {
            mio::log_error("Transition from {} to {} with status {} caused negative value.", r.from, r.to, r.status);
        }
        m_simulations[r.to].get_result().get_last_value()[static_cast<Eigen::Index>(r.status)] += 1;
    }
    /// @brief compute the current value of a transition rate
    ScalarType compute_rate(const TransitionRate<Status>& r) const
    {
        return r(m_simulations[r.from].get_result().get_last_value());
    }
    /// @brief draw time until the next event occurs
    ScalarType draw_waiting_time()
    {
        // compute the current cumulative transition rate
        ScalarType ctr = 0; // Lambda
        for (auto rate : m_model.get_transition_rates()) {
            ctr += compute_rate(rate);
        }
        // draw the normalized waiting time
        ScalarType nwt = mio::ExponentialDistribution<ScalarType>::get_instance()(1.0); // tau'
        return nwt / ctr;
    }
    /// @brief dynamically set m_dt
    void update_dt(ScalarType dt_max)
    {
        m_dt = dt_max;
        for (auto&& sim : m_simulations) {
            if (sim.get_dt() < m_dt) {
                m_dt = sim.get_dt();
            }
        }
    }
    /// @brief updates m_results to current time m_t0
    void update_results()
    {
        auto first_new_result_index = m_results.get_num_time_points();
        // decide which time points to add
        // choose maximum time resolution
        Eigen::Index max = 0;
        size_t max_id    = 0;
        for (size_t sim_id = 0; sim_id < m_simulations.size(); sim_id++) {
            auto& result = m_simulations[sim_id].get_result();
            if (max < result.get_num_time_points()) {
                max = result.get_num_time_points();
                // keep track on which results to copy
                max_id = sim_id;
            }
        }
        // add all new time points (after m_result_iterator)
        auto range = m_simulations[max_id].get_result().get_times();
        for (auto time = range.begin() + m_result_iterator[max_id] + 1; time != range.end(); time++) {
            m_results.add_time_point(*time);
        }
        // assign interpolated values
        for (size_t sim_id = 0; sim_id < m_simulations.size(); sim_id++) {
            auto& result = m_simulations[sim_id].get_result(); // result alias
            auto& i      = m_result_iterator[sim_id]; // iterator alias
            auto j       = first_new_result_index; // copy by value of first_new_result_index
            assert(m_results.get_time(j - 1) == result.get_time(i));
            // set all but the last new time points
            while (j < m_results.get_num_time_points() && i < result.get_num_time_points() - 1) {
                if (m_results.get_time(j) >= result.get_time(i + 1)) {
                    i++;
                }
                else if (m_results.get_time(j) < result.get_time(i)) {
                    // TODO: remove this else if
                    assert(false && "This should NEVER HAPPEN :(");
                    j++;
                }
                else {
                    // result.get_time(i) <= m_results.get_time(j) < result.get_time(i + 1);
                    ScalarType t =
                        (m_results.get_time(j) - result.get_time(i)) / (result.get_time(i + 1) - result.get_time(i));
                    assert(0 <= t && t < 1);
                    // assign value at the correct segment
                    m_results.get_value(j).segment(sim_id * Status::Count, Status::Count) =
                        (1 - t) * result.get_value(i) + t * result.get_value(i + 1);
                    j++;
                }
            }
            // set last time point (all simulations stop at exactly m_t0)
            m_results.get_last_value().segment(sim_id * Status::Count, Status::Count) = result.get_last_value();
            //i++; // so that m_result_iterator has the index of result.end()
        }
    }
    ScalarType m_dt, m_t0;
    const PDMModel<Status>& m_model;
    std::vector<Eigen::Index> m_result_iterator;
    mio::TimeSeries<ScalarType> m_results;
    std::vector<mio::Simulation<PDMNode<Status>>> m_simulations;
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
    using Model  = PDMNode<InfectionState>;
    using Status = Model::Compartments;
    /*** CONFIG ***/
    mio::set_log_level(mio::LogLevel::warn);
    ScalarType kappa = 0.001;
    /*** END CONFIG ***/

    PDMModel<Status> pdmm({{950, 50, 0}, {1000, 0, 0}},
                          {{Status::S, Status::I, 0.3, Order::second, 0},
                           {Status::I, Status::R, 0.1, Order::first, 0},
                           {Status::S, Status::I, 1, Order::second, 1},
                           {Status::I, Status::R, 0.08, Order::first, 1}},
                          {{Status::S, 0, 1, 0.5 * kappa},
                           {Status::I, 0, 1, 0.1 * kappa},
                           {Status::R, 0, 1, 0.1 * kappa},
                           {Status::S, 1, 0, 0.1 * kappa},
                           {Status::I, 1, 0, 0.02 * kappa},
                           {Status::R, 1, 0, 0.2 * kappa}});
    auto sim = Simulation<Status>(pdmm, 0, 0.1);
    sim.advance(100);

    std::cout << "Global Model " << 1 << "/2\n";
    print_to_terminal(sim.get_result(), {"S", "I", "R", "S", "I", "R"});
    std::cout << "Global Model " << 2 << "/2\n";

    return 0;
}