#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/quad_well.h"
#include "examples/hybrid.h"
#include "memilio/utils/custom_index_array.h"
#include "memilio/utils/logging.h"
#include "memilio/utils/random_number_generator.h"
#include "mpm/abm.h"
#include "mpm/model.h"
#include "mpm/pdmm.h"
#include "mpm/region.h"

#include <vector>

const qw::MetaregionSampler pos_rng{{-2, -2}, {0, 0}, {2, 2}, 0.3};

enum InfStates
{
    S,
    I,
    R,
    Count
};

template <>
void mio::convert_model(const Simulation<mio::mpm::ABM<QuadWellModel<InfStates>>>&,
                        mio::mpm::PDMModel<4, QuadWellModel<InfStates>::Status>&)
{
    DEBUG("convert_model abm->pdmm")
}

template <>
void mio::convert_model(const Simulation<mio::mpm::PDMModel<4, QuadWellModel<InfStates>::Status>>& a,
                        mio::mpm::ABM<QuadWellModel<InfStates>>& b)
{
    DEBUG("convert_model pdmmm->abm")
    using Status = QuadWellModel<InfStates>::Status;

    auto values = a.get_result().get_last_value().eval();

    for (auto& p : b.populations) {
        auto i     = mio::DiscreteDistribution<size_t>::get_instance()(values);
        values[i]  = std::max(0., values[i] - 1);
        p.status   = static_cast<Status>(i % static_cast<size_t>(Status::Count));
        auto well  = i / static_cast<size_t>(Status::Count);
        p.position = pos_rng(well);
    }
}

int main()
{
    mio::set_log_level(mio::LogLevel::warn);

    using namespace mio::mpm;
    using Model            = ABM<QuadWellModel<InfStates>>;
    using Status           = Model::Status;
    const unsigned regions = 4;
    size_t n_agents        = 4000;

    std::vector<Model::Agent> agents(n_agents);

    // std::ofstream apre(mio::base_dir() + "agents_pre.txt"), apost(mio::base_dir() + "agents_post.txt");

    const std::vector<double> pop_dist{0.99, 0.01, 0.0};
    auto& sta_rng = mio::DiscreteDistribution<int>::get_instance();
    int ctr       = 0;
    for (auto& a : agents) {
        a.position = pos_rng(ctr);
        // apre << a.position[0] << " " << a.position[1] << "\n";
        if (ctr == 0) {
            a.status = static_cast<Status>(sta_rng(pop_dist));
        }
        else {
            a.status = Status::S;
        }
        ctr++;
        ctr %= regions;
    }

    std::vector<AdoptionRate<Status>> adoption_rates = {
        {Status::S, Status::I, 0, 0.3, {Status::I}, {1}}, {Status::I, Status::R, 0, 0.1},
        {Status::S, Status::I, 1, 0.3, {Status::I}, {1}}, {Status::I, Status::R, 1, 0.1},
        {Status::S, Status::I, 2, 0.3, {Status::I}, {1}}, {Status::I, Status::R, 2, 0.1},
        {Status::S, Status::I, 3, 0.5, {Status::I}, {1}}, {Status::I, Status::R, 3, 0.08}};

    Model abm(agents, adoption_rates, 0.4, 0.5);

    //mio::mpm::print_to_terminal(mio::simulate(0, 100, 0.1, model), {"S", "I", "R", "S", "I", "R", "S", "I", "R", "S", "I", "R"});

    PDMModel<regions, Status> pdmm;
    std::vector<std::vector<ScalarType>> populations = {{990, 10, 0}, {1000, 0, 0}, {1000, 0, 0}, {1000, 0, 0}};
    pdmm.parameters.get<AdoptionRates<Status>>()     = adoption_rates;

    pdmm.parameters.get<TransitionRates<Status>>().reserve(Status::Count * regions * (regions - 1));
    for (int from = 0; from < regions; from++) {
        for (int to = 0; to < regions; to++) {
            if (from == to)
                continue;
            for (int s = 0; s < (int)Status::Count; s++) {
                if (from == regions - to - 1) {
                    pdmm.parameters.get<TransitionRates<Status>>().push_back(
                        {(Status)s, (Region)from, (Region)to, regions * 0.00026 / n_agents});
                }
                else if (from != to) {
                    pdmm.parameters.get<TransitionRates<Status>>().push_back(
                        {(Status)s, (Region)from, (Region)to, regions * 1.31202 / n_agents});
                }
            }
        }
    }

    for (size_t k = 0; k < regions; k++) {
        for (int i = 0; i < static_cast<size_t>(Status::Count); i++) {
            pdmm.populations[{static_cast<Region>(k), static_cast<Status>(i)}] = populations[k][i];
        }
    }

    // Eigen::Matrix<double, 4, 4> avg_transitions = Eigen::Matrix<double, 4, 4>::Zero();
    // const int n_samples                         = 1000;

    // for (int i = 0; i < n_samples; i++) {
    mio::HybridSimulation<mio::mpm::ABM<QuadWellModel<Status>>, mio::mpm::PDMModel<regions, Status>> sim(abm, pdmm,
                                                                                                         0.5);

    sim.advance(100, [](bool b, const auto& results) {
        const int critical_num_infections = 10;

        bool use_base = true; // some I comps are subcritical
        bool use_sec  = true; // all I comps are critical

        for (size_t i = 0; i < regions; i++) {
            if (results.get_last_value()[mio::flatten_index<mio::Index<Region, Status>>(
                    {Region(i), Status::I}, {Region(regions), Status::Count})] < critical_num_infections) {
                use_base &= true;
                use_sec = false;
            }
            else {
                use_base = false;
                use_sec &= true;
            }
        }

        if (use_base == !use_sec)
            return use_base;
        else
            return b;
    });
    //     avg_transitions += sim.get_base_simulation().get_model().total_transitions / n_samples;
    // }

    // for (const auto& a : sim.get_base_simulation().get_model().populations) {
    //     apost << a.position[0] << " " << a.position[1] << "\n";
    // }

    FILE* file = fopen((mio::base_dir() + "output.txt").c_str(), "w");
    print_to_file(file, sim.get_result(), {"S", "I", "R", "S", "I", "R", "S", "I", "R", "S", "I", "R"});
    fclose(file);

    return 0;
}