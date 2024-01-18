#include "hybrid_paper/library/infection_state.h"
#include "hybrid_paper/library/initialization.h"
#include "hybrid_paper/library/model_setup.h"
#include "memilio/utils/random_number_generator.h"
#include "mpm/abm.h"
#include "hybrid_paper/library/potentials/commuting_potential.h"
#include "mpm/pdmm.h"
#include <cmath>
#include <string>

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()
#define PRECISION 17

#define restart_timer(timer, description)                                                                              \
    {                                                                                                                  \
        TIME_TYPE new_time = TIME_NOW;                                                                                 \
        std::cout << "\r" << description << " :: " << PRINTABLE_TIME(new_time - timer) << std::endl << std::flush;     \
        timer = new_time;                                                                                              \
    }

void run_simulation()
{
    using Status             = mio::mpm::paper::InfectionState;
    using Region             = mio::mpm::Region;
    using ABM                = mio::mpm::ABM<CommutingPotential<StochastiK, Status>>;
    using PDMM               = mio::mpm::PDMModel<8, Status>;
    const size_t num_regions = 8;
    const int focus_region   = 3;

    mio::Date start_date             = mio::Date(2021, 3, 1);
    double t_Exposed                 = 4;
    double t_Carrier                 = 2.67;
    double t_Infected                = 5.03;
    double mu_C_R                    = 0.29;
    double transmission_rate         = 0.25;
    double mu_I_D                    = 0.00476;
    double scaling_factor_trans_rate = 1.0; /// 6.48;
    double contact_radius            = 50;
    double persons_per_agent         = 300;
    double tmax                      = 30;
    double dt                        = 0.1;

    std::vector<int> regions        = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};

    Eigen::MatrixXi metaregions;
    {
        const auto fname = mio::base_dir() + "metagermany.pgm";
        std::ifstream ifile(fname);
        if (!ifile.is_open()) {
            mio::log(mio::LogLevel::critical, "Could not open file {}", fname);
            std::abort();
        }
        else {
            metaregions = mio::mpm::read_pgm_raw(ifile).first;
            ifile.close();
        }
    }

    WeightedGradient wg(mio::base_dir() + "potentially_germany_grad.json", mio::base_dir() + "boundary_ids.pgm");
    const std::vector<double> weights(14, 500);
    const std::vector<double> sigmas(num_regions, 10);
    wg.apply_weights(weights);

    const std::vector<int> county_ids   = {233, 228, 242, 223, 238, 232, 231, 229};
    Eigen::MatrixXd reference_commuters = get_transition_matrix(mio::base_dir() + "data/mobility/").value();

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> commute_weights(num_regions, num_regions);
    commute_weights.setZero();
    for (int i = 0; i < num_regions; i++) {
        for (int j = 0; j < num_regions; j++) {
            if (i != j) {
                commute_weights(i, j) = reference_commuters(county_ids[i], county_ids[j]);
            }
        }
        commute_weights(i, i) = populations[i] - commute_weights.row(i).sum();
    }

    std::map<std::tuple<Region, Region>, double> transition_factors{
        {{Region(0), Region(1)}, 0.000958613}, {{Region(0), Region(2)}, 0.00172736},
        {{Region(0), Region(3)}, 0.0136988},   {{Region(0), Region(4)}, 0.00261568},
        {{Region(0), Region(5)}, 0.000317227}, {{Region(0), Region(6)}, 0.000100373},
        {{Region(0), Region(7)}, 0.00012256},  {{Region(1), Region(0)}, 0.000825387},
        {{Region(1), Region(2)}, 0.00023648},  {{Region(1), Region(3)}, 0.0112213},
        {{Region(1), Region(4)}, 0.00202101},  {{Region(1), Region(5)}, 0.00062912},
        {{Region(1), Region(6)}, 0.000201067}, {{Region(1), Region(7)}, 0.000146773},
        {{Region(2), Region(0)}, 0.000712533}, {{Region(2), Region(1)}, 0.000102613},
        {{Region(2), Region(3)}, 0.00675979},  {{Region(2), Region(4)}, 0.00160171},
        {{Region(2), Region(5)}, 0.000175467}, {{Region(2), Region(6)}, 0.00010336},
        {{Region(2), Region(7)}, 6.21867e-05}, {{Region(3), Region(0)}, 0.00329632},
        {{Region(3), Region(1)}, 0.00322347},  {{Region(3), Region(2)}, 0.00412565},
        {{Region(3), Region(4)}, 0.0332566},   {{Region(3), Region(5)}, 0.00462197},
        {{Region(3), Region(6)}, 0.00659424},  {{Region(3), Region(7)}, 0.00255147},
        {{Region(4), Region(0)}, 0.000388373}, {{Region(4), Region(1)}, 0.000406827},
        {{Region(4), Region(2)}, 0.000721387}, {{Region(4), Region(3)}, 0.027394},
        {{Region(4), Region(5)}, 0.00127328},  {{Region(4), Region(6)}, 0.00068224},
        {{Region(4), Region(7)}, 0.00104491},  {{Region(5), Region(0)}, 0.00013728},
        {{Region(5), Region(1)}, 0.000475627}, {{Region(5), Region(2)}, 0.00010688},
        {{Region(5), Region(3)}, 0.00754293},  {{Region(5), Region(4)}, 0.0034704},
        {{Region(5), Region(6)}, 0.00210027},  {{Region(5), Region(7)}, 0.000226667},
        {{Region(6), Region(0)}, 7.264e-05},   {{Region(6), Region(1)}, 0.0001424},
        {{Region(6), Region(2)}, 9.55733e-05}, {{Region(6), Region(3)}, 0.00921109},
        {{Region(6), Region(4)}, 0.0025216},   {{Region(6), Region(5)}, 0.00266944},
        {{Region(6), Region(7)}, 0.00156053},  {{Region(7), Region(0)}, 7.81867e-05},
        {{Region(7), Region(1)}, 0.0001024},   {{Region(7), Region(2)}, 8.256e-05},
        {{Region(7), Region(3)}, 0.00833152},  {{Region(7), Region(4)}, 0.00393717},
        {{Region(7), Region(5)}, 0.000354987}, {{Region(7), Region(6)}, 0.00055456}};

    mio::mpm::paper::ModelSetup<ABM::Agent> setup(
        t_Exposed, t_Carrier, t_Infected, transmission_rate, mu_C_R, mu_I_D, start_date, regions, populations,
        persons_per_agent, metaregions, tmax, dt, commute_weights, wg, sigmas, contact_radius, transition_factors);

    // std::vector<ABM::Agent> agents_focus_region;
    // std::copy_if(setup.agents.begin(), setup.agents.end(), std::back_inserter(agents_focus_region), [focus_region](ABM::Agent a) {
    //     return a.region == focus_region;
    // });

    ABM abm   = setup.create_abm<ABM>();
    PDMM pdmm = setup.create_pdmm<PDMM>();

    pdmm.populations.array().setZero();

    auto& transition_rates = pdmm.parameters.get<mio::mpm::TransitionRates<Status>>();
    for (auto tr = transition_rates.begin(); tr < transition_rates.end(); ++tr) {
        if (tr->from == Region(focus_region)) {
            tr = transition_rates.erase(tr);
        }
    }
    auto& adoption_rates = pdmm.parameters.get<mio::mpm::AdoptionRates<Status>>();
    for (auto ar = adoption_rates.begin(); ar < adoption_rates.end(); ++ar) {
        if (ar->region == Region(focus_region)) {
            ar = adoption_rates.erase(ar);
        }
    }

    auto simABM  = mio::Simulation<ABM>(abm, 0.0, dt);
    auto simPDMM = mio::Simulation<PDMM>(pdmm, 0.0, dt);

    // set how many commuters enter the focus region each day
    Eigen::VectorXd posteriori_commute_weight = setup.k_provider.metaregion_commute_weights.col(focus_region);
    posteriori_commute_weight[focus_region]   = 0;

    for (double t = -dt;; t = std::min(t + dt, tmax)) {
        std::cerr << t << " / " << tmax << std::endl << std::flush;
        simABM.advance(t);
        simPDMM.advance(t);
        { //move agents from abm to pdmm
            const auto pop     = simPDMM.get_model().populations;
            auto simPDMM_state = simPDMM.get_result().get_last_value();
            auto& agents       = simABM.get_model().populations;
            auto itr           = agents.begin();
            while (itr != agents.end()) {
                if (itr->region != focus_region) {
                    simABM.get_result().get_last_value()[simPDMM.get_model().populations.get_flat_index(
                        {Region(focus_region), itr->status})] -= 1;
                    simPDMM_state[pop.get_flat_index({Region(itr->region), itr->status})] += 1;
                    itr = agents.erase(itr);
                }
                else {
                    itr++;
                }
            }
        }
        { //move agents from pdmm to abm
            auto pop = simPDMM.get_result().get_last_value();
            for (int i = 0; i < (int)Status::Count; i++) {
                // auto p = simPDMM_state[pop.get_flat_index({focus_region, (InfectionState)i})];
                auto& agents = pop[simPDMM.get_model().populations.get_flat_index({Region(focus_region), (Status)i})];
                for (; agents > 0; agents -= 1) {
                    const double daytime = t - std::floor(t);
                    if (daytime < 13. / 24.) {
                        const size_t commuting_origin =
                            mio::DiscreteDistribution<size_t>::get_instance()(posteriori_commute_weight);
                        const double t_return =
                            std::floor(t) +
                            mio::ParameterDistributionNormal(13.0 / 24.0, 23.0 / 24.0, 18.0 / 24.0).get_rand_sample();
                        simABM.get_model().populations.push_back(
                            {setup.k_provider.metaregion_sampler(focus_region), (Status)i, focus_region, true,
                             setup.k_provider.metaregion_sampler(commuting_origin), t_return, 0});
                    }
                    else {
                        simABM.get_model().populations.push_back(
                            {setup.k_provider.metaregion_sampler(focus_region), (Status)i, focus_region, false});
                    }
                }
            }
        }
        if (t >= tmax)
            break;
    }
}

int main(int argc, char** argv)
{
    mio::set_log_level(mio::LogLevel::warn);
    run_simulation();
    return 0;
}