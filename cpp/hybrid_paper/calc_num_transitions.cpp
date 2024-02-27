#include "hybrid_paper/library/infection_state.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "models/mpm/smm.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "hybrid_paper/library/model_setup.h"
#include <cstddef>
#include <map>

int main(int argc, char** argv)
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status            = mio::mpm::paper::InfectionState;
    using ABMKedaechtnislos = mio::mpm::ABM<CommutingPotential<Kedaechtnislos, Status>>;
    using ABMReal           = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    using PDMM              = mio::mpm::PDMModel<8, Status>;
    using SMM               = mio::mpm::SMModel<8, Status>;
    using Region            = mio::mpm::Region;

    mio::mpm::paper::ModelSetup<ABMKedaechtnislos::Agent> setup;
    setup.adoption_rates.clear();

    //set all agents status to S
    for (auto& agent : setup.agents) {
        agent.status = Status::S;
    }
    for (int k = 0; k < 8; k++) {
        setup.pop_dists_scaled[k][0] =
            std::accumulate(setup.pop_dists_scaled[k].begin(), setup.pop_dists_scaled[k].end(), 0.0);
        for (int s = 1; s < (int)Status::Count; s++) {
            setup.pop_dists_scaled[k][s] = 0;
        }
    }

    //delete transitions rates for status E to D
    auto& transition_rates = setup.transition_rates;
    for (auto tr = transition_rates.begin(); tr < transition_rates.end();) {
        if (tr->status != Status::S) {
            tr = transition_rates.erase(tr);
        }
        else {
            ++tr;
        }
    }

    ABMKedaechtnislos abm_kedaechtnislos = setup.create_abm<ABMKedaechtnislos>();
    ABMReal abm_real                     = setup.create_abm<ABMReal>();
    PDMM pdmm                            = setup.create_pdmm<PDMM>();
    SMM smm;
    smm.populations                                         = pdmm.populations;
    smm.parameters.get<mio::mpm::AdoptionRates<Status>>()   = pdmm.parameters.get<mio::mpm::AdoptionRates<Status>>();
    smm.parameters.get<mio::mpm::TransitionRates<Status>>() = pdmm.parameters.get<mio::mpm::TransitionRates<Status>>();

    double tmax = 50.0;

    mio::Simulation<ABMKedaechtnislos> sim_kedaechtnislos(abm_kedaechtnislos, 0, 0.1);
    mio::Simulation<ABMReal> sim_real(abm_real, 0, 0.1);
    mio::Simulation<PDMM> sim_pdmm(pdmm, 0, 0.1);
    //mio::Simulation<SMM> sim_smm(smm, 0, 0.1);
    (reinterpret_cast<mio::ControlledStepperWrapper<boost::numeric::odeint::runge_kutta_cash_karp54>*>(
         &sim_pdmm.get_integrator()))
        ->set_dt_max(0.1);

    //colums: 0 - ABM Kedaechtnislos, 1 - ABM Real, 2 - PDMM, 3 - Real
    Eigen::MatrixXd transitions = Eigen::MatrixXd::Zero(tmax + 1, 4);

    std::map<std::tuple<Region, Region>, Eigen::MatrixXd> res{
        {{Region(0), Region(1)}, transitions}, {{Region(0), Region(2)}, transitions},
        {{Region(0), Region(3)}, transitions}, {{Region(0), Region(4)}, transitions},
        {{Region(0), Region(5)}, transitions}, {{Region(0), Region(6)}, transitions},
        {{Region(0), Region(7)}, transitions}, {{Region(1), Region(0)}, transitions},
        {{Region(1), Region(2)}, transitions}, {{Region(1), Region(3)}, transitions},
        {{Region(1), Region(4)}, transitions}, {{Region(1), Region(5)}, transitions},
        {{Region(1), Region(6)}, transitions}, {{Region(1), Region(7)}, transitions},
        {{Region(2), Region(0)}, transitions}, {{Region(2), Region(1)}, transitions},
        {{Region(2), Region(3)}, transitions}, {{Region(2), Region(4)}, transitions},
        {{Region(2), Region(5)}, transitions}, {{Region(2), Region(6)}, transitions},
        {{Region(2), Region(7)}, transitions}, {{Region(3), Region(0)}, transitions},
        {{Region(3), Region(1)}, transitions}, {{Region(3), Region(2)}, transitions},
        {{Region(3), Region(4)}, transitions}, {{Region(3), Region(5)}, transitions},
        {{Region(3), Region(6)}, transitions}, {{Region(3), Region(7)}, transitions},
        {{Region(4), Region(0)}, transitions}, {{Region(4), Region(1)}, transitions},
        {{Region(4), Region(2)}, transitions}, {{Region(4), Region(3)}, transitions},
        {{Region(4), Region(5)}, transitions}, {{Region(4), Region(6)}, transitions},
        {{Region(4), Region(7)}, transitions}, {{Region(5), Region(0)}, transitions},
        {{Region(5), Region(1)}, transitions}, {{Region(5), Region(2)}, transitions},
        {{Region(5), Region(3)}, transitions}, {{Region(5), Region(4)}, transitions},
        {{Region(5), Region(6)}, transitions}, {{Region(5), Region(7)}, transitions},
        {{Region(6), Region(0)}, transitions}, {{Region(6), Region(1)}, transitions},
        {{Region(6), Region(2)}, transitions}, {{Region(6), Region(3)}, transitions},
        {{Region(6), Region(4)}, transitions}, {{Region(6), Region(5)}, transitions},
        {{Region(6), Region(7)}, transitions}, {{Region(7), Region(0)}, transitions},
        {{Region(7), Region(1)}, transitions}, {{Region(7), Region(2)}, transitions},
        {{Region(7), Region(3)}, transitions}, {{Region(7), Region(4)}, transitions},
        {{Region(7), Region(5)}, transitions}, {{Region(7), Region(6)}, transitions}};

    for (size_t t = 1; t <= tmax; ++t) {
        sim_kedaechtnislos.advance(t);
        sim_real.advance(t);
        sim_pdmm.advance(t);
        //sim_smm.advance(t);
        for (auto& tr : pdmm.parameters.get<mio::mpm::TransitionRates<Status>>()) {
            auto& m = res.at({tr.from, tr.to});
            m(t, 0) = sim_kedaechtnislos.get_model().number_transitions(tr);
            sim_kedaechtnislos.get_model().number_transitions(tr) = 0;
            m(t, 1)                                               = sim_real.get_model().number_transitions(tr);
            sim_real.get_model().number_transitions(tr)           = 0;
            m(t, 2)                                               = sim_pdmm.get_model().number_transitions(tr);
            sim_pdmm.get_model().number_transitions(tr)           = 0;
            // m(t, 3)                                     = sim_smm.get_model().number_transitions(tr);
            // sim_smm.get_model().number_transitions(tr)  = 0;
            m(t, 3) = (setup.commute_weights(static_cast<size_t>(tr.from), static_cast<size_t>(tr.to)) +
                       setup.commute_weights(static_cast<size_t>(tr.to), static_cast<size_t>(tr.from))) /
                      setup.persons_per_agent;
        }
    }

    //print result
    //std::cout << "from to t ABMKedachtnislos ABMReal PDMM SMM Real" << std::endl;
    for (auto& rate : res) {
        auto from = std::get<0>(rate.first);
        auto to   = std::get<1>(rate.first);
        auto& m   = res.at({from, to});
        for (size_t t = 1; t <= tmax; ++t) {
            std::cout << static_cast<size_t>(from) << " " << static_cast<size_t>(to) << " " << t << " " << m(t, 0)
                      << " " << m(t, 1) << " " << m(t, 2) << " " << m(t, 3) << std::endl;
        }
    }

    return 0;
}