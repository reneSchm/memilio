#include "examples/hybrid.h"
#include "hybrid_paper/library/analyze_result.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/quad_well.h"
#include "hybrid_paper/quad_well/quad_well_setup.h"
#include "memilio/compartments/simulation.h"
#include "memilio/config.h"
#include "memilio/data/analyze_result.h"
#include "memilio/utils/logging.h"
#include "memilio/utils/time_series.h"
#include "mpm/abm.h"
#include "mpm/model.h"
#include "mpm/pdmm.h"
#include "mpm/region.h"
#include <algorithm>
#include <cstdio>
#include <omp.h>
#include <string>
#include <vector>

static qw::MetaregionSampler pos_rng{{-2, -2}, {2, 2}, {2, 2}, 0.3};

class SingleWell
{
public:
    using Status   = mio::mpm::paper::InfectionState;
    using Position = qw::Position;

    struct Agent {
        Position position;
        Status status;
    };

    SingleWell(const std::vector<Agent>& agents, const typename mio::mpm::AdoptionRates<Status>::Type& rates,
               double contact_radius = 0.4, double sigma = 0.4)
        : populations(agents)
        , m_contact_radius(contact_radius)
        , m_sigma(sigma)
    {
        for (auto& agent : populations) {
            assert(is_in_domain(agent.position));
        }
        for (auto& r : rates) {
            m_adoption_rates.emplace(std::forward_as_tuple(r.region, r.from, r.to), r);
            //m_adoption_rates[{r.region, r.from, r.to}] = r;
        }
    }

    inline static constexpr void adopt(Agent& agent, const Status& new_status)
    {
        agent.status = new_status;
    }

    double adoption_rate(const Agent& agent, const Status& new_status) const
    {
        double rate = 0;
        // get the correct adoption rate
        const size_t well = 0;
        auto map_itr      = m_adoption_rates.find({well, agent.status, new_status});
        if (map_itr != m_adoption_rates.end()) {
            const auto& adoption_rate = map_itr->second;
            // calculate the current rate, depending on order
            if (adoption_rate.influences.size() == 0) { // first order adoption
                // contact independant rate
                rate = adoption_rate.factor;
            }
            else { // second order adoption
                // accumulate rate per contact with a status in influences
                size_t num_contacts   = 0;
                ScalarType influences = 0;
                for (auto& contact : populations) {
                    // check if contact is indeed a contact
                    if (is_contact(agent, contact)) {
                        num_contacts++;
                        for (size_t i = 0; i < adoption_rate.influences.size(); i++) {
                            if (contact.status == adoption_rate.influences[i]) {
                                influences += adoption_rate.factors[i];
                            }
                        }
                    }
                }
                // rate = factor * "concentration of contacts with status new_status"
                if (num_contacts > 0) {
                    rate = adoption_rate.factor * (influences / num_contacts);
                }
            }
        }
        // else: no adoption from agent.status to new_status exist
        return rate;
    }

    void move(const double /*t*/, const double dt, Agent& agent)
    {
        Position p = {mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0),
                      mio::DistributionAdapter<std::normal_distribution<double>>::get_instance()(0.0, 1.0)};

        agent.position = agent.position - dt * grad_U(agent.position) + (m_sigma * std::sqrt(dt)) * p;
    }

    Eigen::VectorXd time_point() const
    {
        Eigen::VectorXd val = Eigen::VectorXd::Zero(static_cast<size_t>(Status::Count));
        for (auto& agent : populations) {
            val[static_cast<size_t>(agent.status)] += 1;
        }
        return val;
    }

    std::map<std::tuple<mio::mpm::Region, Status, Status>, mio::mpm::AdoptionRate<Status>>& get_adoption_rates()
    {
        return m_adoption_rates;
    }

    std::vector<Agent> populations;

private:
    static Position grad_U(const Position x)
    {
        // U is a quad well potential
        // U(x0,x1) = (x0^4 + x1^4)/2
        return {2 * x[0] * x[0] * x[0], 2 * x[1] * x[1] * x[1]};
    }

    bool is_contact(const Agent& agent, const Agent& contact) const
    {
        //      test if contact is in the contact radius                     and test if agent and contact are different objects
        return (agent.position - contact.position).norm() < m_contact_radius && (&agent != &contact) &&
               qw::well_index(agent.position) == qw::well_index(contact.position);
    }

    bool is_in_domain(const Position& p) const
    {
        // restrict domain to [-2, 2]^2 where "escaping" is impossible, i.e. it holds x <= grad_U(x) for dt <= 0.1
        return -2 <= p[0] && p[0] <= 2 && -2 <= p[1] && p[1] <= 2;
    }

    std::map<std::tuple<mio::mpm::Region, Status, Status>, mio::mpm::AdoptionRate<Status>> m_adoption_rates;
    double m_contact_radius;
    double m_sigma;
};

using ABM  = mio::mpm::ABM<SingleWell>;
using PDMM = mio::mpm::PDMModel<1, SingleWell::Status>;

template <>
void mio::convert_model(const Simulation<ABM>&, PDMM&)
{
    DEBUG("convert_model abm->pdmm")
}

template <>
void mio::convert_model(const Simulation<PDMM>& a, ABM& b)
{
    DEBUG("convert_model pdmmm->abm")
    using Status = SingleWell::Status;

    auto values = a.get_result().get_last_value().eval();

    for (auto& p : b.populations) {
        auto i     = mio::DiscreteDistribution<size_t>::get_instance()(values);
        values[i]  = std::max(0., values[i] - 1);
        p.status   = static_cast<Status>(i % static_cast<size_t>(Status::Count));
        p.position = pos_rng(0);
    }
}

int main()
{
    using Status = SingleWell::Status;

    mio::set_log_level(mio::LogLevel::err);

    const int num_runs          = 10000;
    const size_t n_agents       = 10000;
    const double t0             = 0;
    const double tnpi           = 20;
    const double tmax           = 40;
    const double dt             = 0.1;
    const double dt_switch      = 0.2;
    const double contact_radius = 0.1;
    const double sigma          = 0.5;

    std::vector<SingleWell::Agent> agents(n_agents);
    for (auto& agent : agents) {
        agent.position = pos_rng(0);
    }
    agents[0].status = Status::I;

    QuadWellSetup<SingleWell::Agent> setup(0);
    decltype(setup.adoption_rates) adoption_rates;
    adoption_rates.push_back({Status::S, Status::E, mio::mpm::Region(0), 0.6, {Status::C, Status::I}, {1, 1}});
    adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(0), 1.0 / setup.t_Exposed});
    adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(0), setup.mu_C_R / setup.t_Carrier});
    adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(0), (1 - setup.mu_C_R) / setup.t_Carrier});
    adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(0), (1 - setup.mu_I_D) / setup.t_Infected});
    adoption_rates.push_back({Status::I, Status::D, mio::mpm::Region(0), setup.mu_I_D / setup.t_Infected});

    ABM abm(agents, adoption_rates, contact_radius, sigma);

    PDMM pdmm;
    pdmm.parameters.set<mio::mpm::AdoptionRates<Status>>(adoption_rates);
    pdmm.populations[{mio::mpm::Region(0), Status::S}] = n_agents - 1;
    pdmm.populations[{mio::mpm::Region(0), Status::I}] = 1;

    const auto is_extinction = [](const mio::TimeSeries<double>& result) {
        const auto& v = result.get_last_value();
        return v[(int)Status::E] + v[(int)Status::C] + v[(int)Status::I] <= 1e-14;
    };

    const int survival_threshold = 5;
    // const auto dir               = mio::base_dir() + "results_hybrid/abm_";
    // const auto dir = mio::base_dir() + "results_hybrid/pdmm_";
    const auto dir = mio::base_dir() + "results_hybrid/hybrid_" + std::to_string(survival_threshold) + "_";

    const auto use_base_model = [&is_extinction, survival_threshold](const bool&, const auto& results) {
        // return true;
        // return false;
        const auto& v = results.get_last_value();
        if (is_extinction(results) || (v[(int)Status::E] + v[(int)Status::C] + v[(int)Status::I] > survival_threshold))
            return false;
        else
            return true;
    };

    double time_min = 1e+100, time_mean = 0, time_max = 0;
    double time_min_e = 1e+100, time_mean_e = 0, time_max_e = 0;
    int extinctions = 0;

    std::vector<mio::TimeSeries<double>> results(num_runs, {(int)Status::Count});
#pragma omp parallel for
    for (auto& result : results) {
        auto sim        = mio::HybridSimulation<ABM, PDMM>(abm, pdmm, dt_switch, t0, dt);
        double start_tp = omp_get_wtime();
        sim.advance(tnpi, use_base_model);
        sim.get_base_simulation()
            .get_model()
            .get_adoption_rates()
            .at({mio::mpm::Region(0), Status::S, Status::E})
            .factor *= 0.2;
        sim.get_secondary_simulation().get_model().parameters.get<mio::mpm::AdoptionRates<Status>>()[0].factor *= 0.2;
        sim.advance(tmax, use_base_model);
        double stop_tp = omp_get_wtime();
        result         = sim.get_result();
        // std::cout << result.get_last_value().transpose().cast<int>() << "\n";
#pragma omp critical
        {
            double time = stop_tp - start_tp;
            if (is_extinction(result)) {
                ++extinctions;
                time_min_e = std::min(time, time_min_e);
                time_max_e = std::max(time, time_max_e);
                time_mean_e += time;
            }
            else {
                time_min = std::min(time, time_min);
                time_max = std::max(time, time_max);
                time_mean += time;
            }
        }
        // {
        //     int change = 0;
        //     bool c     = true;
        //     for (Eigen::Index i = result.get_num_time_points() - 1; i >= 0; --i) {
        //         const auto& v = result.get_value(i);
        //         bool b        = v[1] + v[2] + v[3] <= 1e-14;

        //         if (c != b) {
        //             change++;
        //             c = b;
        //         }
        //     }
        //     if (change <= 0) {
        //         std::cerr << "This should never happen\n";
        //         assert(false);
        //     }
        //     if (change == 2) {
        //         change = 0;
        //         mio::mpm::print_to_file(stderr, result, {});
        //     }
        //     if (change >= 3) {
        //         std::cerr << "This should never happen, and it's weird\n";
        //         assert(false);
        //     }
        // }

        result = mio::interpolate_simulation_result(result);
    }
    std::cout << "extinctions: " << (double)extinctions / num_runs * 100 << "%\n";

#pragma omp single
    {
        std::cout << "timing surv: min= " << time_min << " mean= " << (time_mean / (num_runs - extinctions))
                  << " max= " << time_max << "\n"
                  << "timing ext : min= " << time_min_e << " mean= " << (extinctions ? time_mean_e / extinctions : 0)
                  << " max= " << time_max_e << "\n"
                  << "timing all : min= " << std::min(time_min, time_min_e)
                  << " mean= " << ((time_mean + time_mean_e) / num_runs) << " max= " << std::max(time_max, time_max_e)
                  << "\n";

        std::vector<mio::TimeSeries<double>> survival_res, extinction_res;
        for (const auto& result : results) {
            if (is_extinction(result)) {
                extinction_res.push_back(result);
            }
            else {
                survival_res.push_back(result);
            }
        }
        if (survival_res.size() == 0) {
            survival_res.push_back(mio::TimeSeries<double>::zero(1, (int)Status::Count));
        }
        if (extinction_res.size() == 0) {
            extinction_res.push_back(mio::TimeSeries<double>::zero(1, (int)Status::Count));
        }

        const auto print_percentiles = [&dir](const auto& ensemble_result, std::string file_prefix) {
            const auto formatted_result = mio::mpm::get_format_for_percentile_output(ensemble_result, 1);
            auto ensemble_result_p05    = mio::ensemble_percentile(formatted_result, 0.05);
            auto ensemble_result_p25    = mio::ensemble_percentile(formatted_result, 0.25);
            auto ensemble_result_p50    = mio::ensemble_percentile(formatted_result, 0.50);
            auto ensemble_result_p75    = mio::ensemble_percentile(formatted_result, 0.75);
            auto ensemble_result_p95    = mio::ensemble_percentile(formatted_result, 0.95);

            mio::mpm::percentile_output_to_file(ensemble_result_p05, dir + file_prefix + "_p05.txt");
            mio::mpm::percentile_output_to_file(ensemble_result_p25, dir + file_prefix + "_p25.txt");
            mio::mpm::percentile_output_to_file(ensemble_result_p50, dir + file_prefix + "_p50.txt");
            mio::mpm::percentile_output_to_file(ensemble_result_p75, dir + file_prefix + "_p75.txt");
            mio::mpm::percentile_output_to_file(ensemble_result_p95, dir + file_prefix + "_p95.txt");

            auto ensemble_result_mean = std::vector<mio::TimeSeries<double>>{
                std::accumulate(ensemble_result.cbegin(), ensemble_result.cend(),
                                mio::TimeSeries<double>::zero(ensemble_result[0].get_num_time_points(),
                                                              ensemble_result[0].get_num_elements()),
                                mio::mpm::add_time_series)};
            for (Eigen::Index i = 0; i < ensemble_result_mean[0].get_num_time_points(); ++i) {
                ensemble_result_mean[0].get_value(i) *= 1.0 / ensemble_result.size();
            }
            mio::mpm::percentile_output_to_file(ensemble_result_mean, dir + file_prefix + "_mean.txt");
        };

        print_percentiles(results, "combined");
        print_percentiles(extinction_res, "extinction");
        print_percentiles(survival_res, "survival");
    }
    return 0;
}