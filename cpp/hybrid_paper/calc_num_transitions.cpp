#include "hybrid_paper/library/infection_state.h"
#include "models/mpm/abm.h"
#include "models/mpm/pdmm.h"
#include "models/mpm/smm.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "hybrid_paper/munich/munich_setup.h"
#include <cstddef>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status            = mio::mpm::paper::InfectionState;
    using ABMKedaechtnislos = mio::mpm::ABM<CommutingPotential<Kedaechtnislos, Status>>;
    using ABMReal           = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    using PDMM              = mio::mpm::PDMModel<8, Status>;
    using SMM               = mio::mpm::SMModel<8, Status>;
    using Region            = mio::mpm::Region;

    mio::mpm::paper::MunichSetup<ABMReal::Agent> setup;
    setup.adoption_rates.clear();

    //set all agents status to S
    for (auto& agent : setup.agents) {
        agent.status = Status::S;
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

    double tmax     = 50.0;
    size_t num_runs = 100;

    //colums: 0 - ABM Kedaechtnislos, 1 - ABM Real, 2 - PDMM, 3 - Real
    Eigen::MatrixXd transitions = Eigen::MatrixXd::Zero(tmax + 1, 4);

    using result_type = std::map<std::tuple<Region, Region>, std::vector<Eigen::MatrixXd>>;
    result_type res{{{Region(0), Region(1)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(0), Region(2)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(0), Region(3)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(0), Region(4)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(0), Region(5)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(0), Region(6)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(0), Region(7)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(1), Region(0)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(1), Region(2)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(1), Region(3)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(1), Region(4)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(1), Region(5)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(1), Region(6)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(1), Region(7)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(2), Region(0)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(2), Region(1)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(2), Region(3)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(2), Region(4)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(2), Region(5)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(2), Region(6)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(2), Region(7)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(3), Region(0)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(3), Region(1)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(3), Region(2)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(3), Region(4)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(3), Region(5)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(3), Region(6)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(3), Region(7)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(4), Region(0)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(4), Region(1)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(4), Region(2)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(4), Region(3)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(4), Region(5)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(4), Region(6)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(4), Region(7)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(5), Region(0)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(5), Region(1)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(5), Region(2)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(5), Region(3)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(5), Region(4)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(5), Region(6)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(5), Region(7)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(6), Region(0)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(6), Region(1)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(6), Region(2)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(6), Region(3)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(6), Region(4)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(6), Region(5)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(6), Region(7)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(7), Region(0)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(7), Region(1)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(7), Region(2)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(7), Region(3)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(7), Region(4)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(7), Region(5)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)},
                    {{Region(7), Region(6)}, std::vector<Eigen::MatrixXd>(num_runs, transitions)}};

    for (size_t run = 0; run < num_runs; ++run) {
        std::cout << "run " << run << std::endl;
        mio::Simulation<ABMKedaechtnislos> sim_kedaechtnislos(abm_kedaechtnislos, 0, 0.1);
        mio::Simulation<ABMReal> sim_real(abm_real, 0, 0.1);
        mio::Simulation<PDMM> sim_pdmm(pdmm, 0, 0.1);

        for (size_t t = 1; t <= tmax; ++t) {
            //sim_kedaechtnislos.advance(t);
            sim_real.advance(t);
            sim_pdmm.advance(t);
            //sim_smm.advance(t);
            for (auto& tr : pdmm.parameters.get<mio::mpm::TransitionRates<Status>>()) {
                auto& m = res.at({tr.from, tr.to})[run];
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
    }

    std::vector<double> percentiles{0.05, 0.25, 0.5, 0.75, 0.95};

    std::cout << "from to t ABMReal PDMM Real" << std::endl;
    for (auto p : percentiles) {
        std::string fname = mio::base_dir() + "cpp/" + std::to_string(p) + "_num_transitions_scale1.txt";
        std::ofstream ofile(fname);
        //calculate percentiles matrix
        for (auto& rate : res) {
            auto from                         = std::get<0>(rate.first);
            auto to                           = std::get<1>(rate.first);
            Eigen::MatrixXd percentile_matrix = Eigen::MatrixXd::Zero(tmax + 1, 4);
            auto& run_matrix                  = res.at({from, to});
            for (size_t time = 0; time <= tmax; ++time) {
                percentile_matrix(time, 3) = run_matrix[0](time, 3);
                percentile_matrix(time, 0) = time;
                std::vector<double> sorted_elems_abm(num_runs);
                std::vector<double> sorted_elems_pdmm(num_runs);
                for (size_t run = 0; run < num_runs; ++run) {
                    sorted_elems_abm[run]  = run_matrix[run](time, 1);
                    sorted_elems_pdmm[run] = run_matrix[run](time, 2);
                }
                std::sort(sorted_elems_abm.begin(), sorted_elems_abm.end());
                std::sort(sorted_elems_pdmm.begin(), sorted_elems_pdmm.end());
                percentile_matrix(time, 1) = sorted_elems_abm[static_cast<size_t>(num_runs * p)];
                percentile_matrix(time, 2) = sorted_elems_pdmm[static_cast<size_t>(num_runs * p)];
            }
            for (size_t t = 1; t <= tmax; ++t) {
                ofile << static_cast<size_t>(from) << " " << static_cast<size_t>(to) << " " << percentile_matrix(t, 0)
                      << " " << percentile_matrix(t, 1) << " " << percentile_matrix(t, 2) << " "
                      << percentile_matrix(t, 3) << std::endl;
            }
        }
        ofile.close();
    }

    std::cout << "done\n";

    return 0;
}