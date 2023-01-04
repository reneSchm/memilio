#include "memilio/compartments/simulation.h"
#include "memilio/config.h"
#include "memilio/epidemiology/populations.h"
#include "memilio/compartments/compartmentalmodel.h"
#include "memilio/math/adapt_rk.h"
#include "memilio/math/eigen.h"
#include "memilio/math/integrator.h"
#include "memilio/utils/custom_index_array.h"
#include "memilio/utils/logging.h"
#include "memilio/utils/parameter_distributions.h"
#include "memilio/utils/parameter_set.h"
#include "memilio/utils/random_number_generator.h"
#include "memilio/utils/time_series.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

enum class Order
{
    first,
    second
};

enum class InfectionState
{
    S,
    I,
    R,
    Count
};

struct Metapopulation : public mio::Index<Metapopulation> {
    Metapopulation(const size_t num_metapopulations)
        : Index<Metapopulation>(num_metapopulations)
    {
    }
};

template <class Status>
struct AdoptionRate {
    Status from; // i
    Status to; // j
    ScalarType factor; // gammahat_{ij}^k
    Order order; // first or second
    Metapopulation location; // k
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
    Metapopulation from; // k
    Metapopulation to; // l
    ScalarType factor; // lambda_i^{kl}
};

template <class Status>
struct TransitionRates {
    using Type = std::vector<TransitionRate<Status>>;
    const static std::string name()
    {
        return "TransitionRates";
    }
};

template <size_t mps, class Status>
class MetapopulationModel
    : public mio::CompartmentalModel<Status, mio::Populations<Metapopulation, Status>,
                                     mio::ParameterSet<AdoptionRates<Status>, TransitionRates<Status>>>
{
    using Base = mio::CompartmentalModel<Status, mio::Populations<Metapopulation, Status>,
                                         mio::ParameterSet<AdoptionRates<Status>, TransitionRates<Status>>>;

public:
    MetapopulationModel()
        : Base(typename Base::Populations({static_cast<Metapopulation>(mps), Status::Count}, 0.),
               typename Base::ParameterSet())
    {
    }

    void get_derivatives(Eigen::Ref<const Eigen::VectorXd> x, Eigen::Ref<const Eigen::VectorXd>, ScalarType t,
                         Eigen::Ref<Eigen::VectorXd> dxdt) const override
    {
        const auto& params = this->parameters;
        const auto& pop    = this->populations;
        for (const auto& rate : params.template get<AdoptionRates<Status>>()) {
            const auto source = pop.get_flat_index({rate.location, rate.from});
            const auto target = pop.get_flat_index({rate.location, rate.to});
            const auto change = evaluate(rate, x);
            dxdt[source] -= change;
            dxdt[target] += change;
        }
    }

    static constexpr ScalarType evaluate(const AdoptionRate<Status>& rate, const Eigen::VectorXd& x)
    {
        const auto source = flat_index({rate.location, rate.from});
        const auto target = flat_index({rate.location, rate.to});
        // calculate rate depending on order
        if (rate.order == Order::first) {
            return rate.factor * x[source];
        }
        else {
            // calculate current size of metapopulation
            const ScalarType N =
                x.segment(rate.location.get() * static_cast<size_t>(Status::Count), static_cast<size_t>(Status::Count))
                    .sum();
            return rate.factor * x[source] * x[target] / ((N > 0) ? N : 1);
        }
    }

    static constexpr ScalarType evaluate(const TransitionRate<Status>& rate, const Eigen::VectorXd& x)
    {
        const auto source = flat_index({rate.from, rate.status});
        return rate.factor * x[source];
    }

private:
    static constexpr Eigen::Index flat_index(const typename Base::Populations::Index& indices)
    {
        return mio::details::flatten_index<0>(indices, {static_cast<Metapopulation>(mps), Status::Count}).first;
    }
};

template <size_t mps, class Status>
class PDMModel : public MetapopulationModel<mps, Status>
{
public:
    using MetapopulationModel<mps, Status>::MetapopulationModel;
};

template <class Status, size_t mps>
class mio::Simulation<PDMModel<mps, Status>>
{
    using Sim = mio::Simulation<MetapopulationModel<mps, Status>>;

public:
    using Model = PDMModel<mps, Status>;

    /**
     * @brief setup the simulation with an ODE solver
     * @param[in] model: An instance of a compartmental model
     * @param[in] t0 start time
     * @param[in] dt initial step size of integration
     */
    Simulation(Model const& model, double t0 = 0., double dt = 0.1)
        : m_dt(dt)
        , m_t0(t0)
        , m_model(model)
        , m_transition_rates(model.parameters.template get<TransitionRates<Status>>())
        , m_simulation(Sim(model, t0, dt))
    {
        assert(std::all_of(m_transition_rates.begin(), m_transition_rates.end(), [](auto&& r) {
            return static_cast<size_t>(r.from) < mps && static_cast<size_t>(r.to) < mps;
        }));
    }

    /**
     * @brief advance simulation to tmax
     * tmax must be greater than get_result().get_last_time_point()
     * @param tmax next stopping point of simulation
     */
    Eigen::Ref<Eigen::VectorXd> advance(ScalarType tmax, ScalarType dt_max = std::numeric_limits<ScalarType>::max())
    {
        // determine how long we wait until the next transition occurs
        ScalarType waiting_time = draw_waiting_time();
        ScalarType remaining_time;
        // iterate time by increments of m_dt
        while (m_t0 < tmax) {
            ScalarType dt  = std::min({m_dt, tmax - m_t0});
            remaining_time = dt; // xi * Delta-t
            // check if one or more transitions occur during this time step
            if (waiting_time < dt) { // (at least one) event occurs
                std::vector<ScalarType> rates(m_transition_rates.size()); // lambda_m (for all m)
                // perform transition(s)
                do {
                    m_t0 += waiting_time; // event time t**
                    // advance all locations
                    m_simulation.advance(m_t0);
                    // compute current transition rates
                    std::transform(m_transition_rates.begin(), m_transition_rates.end(), rates.begin(),
                                   [&](auto&& rate) {
                                       // we should normalize each term by dividing by the cumulative transition rate,
                                       // but the DiscreteDistribution below effectively does that for us
                                       return m_model.evaluate(rate, get_result().get_last_value());
                                   });
                    // draw transition event, then execute it
                    size_t event = mio::DiscreteDistribution<size_t>::get_instance()(rates);
                    perform_transition(event);
                    // draw new waiting time
                    remaining_time -= waiting_time;
                    waiting_time = draw_waiting_time();
                    // repeat, if another event occurs in the remaining time interval
                } while (waiting_time < remaining_time);
            }
            else { // no event occurs
                // reduce waiting time for the next step
                waiting_time -= dt;
            }
            // advance time
            m_t0 += remaining_time;
            // advance all locations
            m_simulation.advance(m_t0);
            m_dt = std::min(m_simulation.get_dt(), dt_max);
        }
        return get_result().get_last_value();
    }

    /**
     * @brief set the core integrator used in the simulation
     */
    void set_integrator(std::shared_ptr<IntegratorCore> integrator)
    {
        m_simulation.set_integrator(integrator);
    }

    /**
     * @brief get_integrator
     * @return reference to the core integrator used in the simulation
     */
    IntegratorCore& get_integrator()
    {
        return m_simulation.get_integrator();
    }

    /**
     * @brief get_integrator
     * @return reference to the core integrator used in the simulation
     */
    IntegratorCore const& get_integrator() const
    {
        return m_simulation.get_integrator();
    }

    /**
     * @brief get_result returns the final simulation result
     * @return a TimeSeries to represent the final simulation result
     */
    TimeSeries<ScalarType>& get_result()
    {
        return m_simulation.get_result();
    }

    /**
     * @brief get_result returns the final simulation result
     * @return a TimeSeries to represent the final simulation result
     */
    const TimeSeries<ScalarType>& get_result() const
    {
        return m_simulation.get_result();
    }

    /**
     * @brief returns the simulation model used in simulation
     */
    const Model& get_model() const
    {
        return m_simulation.get_model();
    }

    /**
     * @brief returns the simulation model used in simulation
     */
    Model& get_model()
    {
        return m_simulation.get_model();
    }

    double& get_dt()
    {
        return m_simulation.get_dt();
    }

    const double& get_dt() const
    {
        return m_simulation.get_dt();
    }

private:
    /// @brief draw time until the next event occurs
    ScalarType draw_waiting_time()
    {
        // compute the current cumulative transition rate
        ScalarType ctr = 0; // Lambda
        for (auto rate : m_transition_rates) {
            ctr += m_model.evaluate(rate, get_result().get_last_value());
        }
        // draw the normalized waiting time
        ScalarType nwt = mio::ExponentialDistribution<ScalarType>::get_instance()(1.0); // tau'
        // "un-normalize", i.e. scale nwt by time = 1/rate
        return nwt / ctr;
    }
    /// @brief perform the given transition event, moving a single "agent"
    void perform_transition(size_t event)
    {
        // get the rate corresponding to the event
        const auto& rate = m_transition_rates[event];
        // transitioning one person at the current time
        auto& value = get_result().get_last_value()[m_model.populations.get_flat_index({rate.from, rate.status})];
        value -= 1;
        if (value < 0) {
            std::cerr << "Transition from " << static_cast<size_t>(rate.from) << " to " << static_cast<size_t>(rate.to)
                      << " with status " << static_cast<size_t>(rate.status) << " caused negative value.\n";
            mio::log_error("Transition from {} to {} with status {} caused negative value.",
                           static_cast<size_t>(rate.from), static_cast<size_t>(rate.to),
                           static_cast<size_t>(rate.status));
        }
        get_result().get_last_value()[m_model.populations.get_flat_index({rate.to, rate.status})] += 1;
    }

    ScalarType m_dt, m_t0;
    const PDMModel<mps, Status>& m_model;
    const typename TransitionRates<Status>::Type& m_transition_rates;
    Sim m_simulation;
};

template <class SC>
void print_to_terminal(const mio::TimeSeries<SC>& results, const std::vector<std::string>& state_names)
{
    printf("%-16s   ", "Time");
    for (size_t k = 0; k < results.get_num_elements(); k++) {
        if (k < state_names.size())
            printf(" %-16s ", state_names[k].data()); // print underlying char*
        else
            printf(" %-16s ", ("#" + std::to_string(k + 1)).data());
    }
    auto num_points = static_cast<size_t>(results.get_num_time_points());
    for (size_t i = 0; i < num_points; i++) {
        printf("\n%16.6f ", results.get_time(i));
        auto res_i = results.get_value(i);
        for (size_t j = 0; j < res_i.size(); j++) {
            printf(" %16.6f ", res_i[j]);
        }
    }
    printf("\n");
}

int main()
{
    const unsigned mps = 2;
    using Model        = PDMModel<mps, InfectionState>;
    using Status       = Model::Compartments;

    Model model;

    /*** CONFIG ***/
    mio::set_log_level(mio::LogLevel::warn);
    ScalarType kappa                                 = 0.001;
    std::vector<std::vector<ScalarType>> populations = {{950, 50, 0}, {1000, 0, 0}};
    model.parameters.get<AdoptionRates<Status>>()    = {{Status::S, Status::I, 0.3, Order::second, 0},
                                                        {Status::I, Status::R, 0.1, Order::first, 0},
                                                        {Status::S, Status::I, 1, Order::second, 1},
                                                        {Status::I, Status::R, 0.08, Order::first, 1}};
    model.parameters.get<TransitionRates<Status>>()  = {{Status::S, 0, 1, 0.5 * kappa},  {Status::I, 0, 1, 0.1 * kappa},
                                                        {Status::R, 0, 1, 0.1 * kappa},  {Status::S, 1, 0, 0.1 * kappa},
                                                        {Status::I, 1, 0, 0.02 * kappa}, {Status::R, 1, 0, 0.2 * kappa}};
    /*** END CONFIG ***/

    for (size_t k = 0; k < mps; k++) {
        for (int i = 0; i < static_cast<size_t>(Status::Count); i++) {
            model.populations[{static_cast<Metapopulation>(k), static_cast<Status>(i)}] = populations[k][i];
        }
    }

    auto result = mio::simulate(0, 100, 0.1, model);

    std::cout << "Global Model " << 1 << "/2\n";
    print_to_terminal(result, {"S", "I", "R", "S", "I", "R"});
    std::cout << "Global Model " << 2 << "/2\n";

    return 0;
}