#include <epidemiology/parameter_studies/parameter_space.h>
#include <epidemiology/parameter_studies/parameter_studies.h>
#include <epidemiology/secir.h>
#include <gtest/gtest.h>
#include <stdio.h>

TEST(ParameterStudies, sample_from_secir_params)
{
    double t0   = 0;
    double tmax = 50;
    double dt   = 0.1;

    double tinc    = 5.2, // R_2^(-1)+R_3^(-1)
        tinfmild   = 6, // 4-14  (=R4^(-1))
        tserint    = 4.2, // 4-4.4 // R_2^(-1)+0.5*R_3^(-1)
        thosp2home = 12, // 7-16 (=R5^(-1))
        thome2hosp = 5, // 2.5-7 (=R6^(-1))
        thosp2icu  = 2, // 1-3.5 (=R7^(-1))
        ticu2home  = 8, // 5-16 (=R8^(-1))
        tinfasy    = 6.2, // (=R9^(-1)=R_3^(-1)+0.5*R_4^(-1))
        ticu2death = 5; // 3.5-7 (=R5^(-1))

    double cont_freq = 0.5, // 0.2-0.75
        alpha        = 0.09, // 0.01-0.16
        beta         = 0.25, // 0.05-0.5
        delta        = 0.3, // 0.15-0.77
        rho          = 0.2, // 0.1-0.35
        theta        = 0.25; // 0.15-0.4

    double nb_total_t0 = 10000, nb_exp_t0 = 100, nb_inf_t0 = 50, nb_car_t0 = 50, nb_hosp_t0 = 20, nb_icu_t0 = 10,
           nb_rec_t0 = 10, nb_dead_t0 = 0;

    int nb_groups = 3;
    double fact   = 1.0 / (double)nb_groups;

    std::vector<epi::SecirParams> params{epi::SecirParams{}};
    epi::ContactFrequencyMatrix contact_freq_matrix{(size_t)nb_groups};
    for (size_t i = 1; i < nb_groups; i++) {
        params.push_back(epi::SecirParams{});
    }

    for (size_t i = 0; i < nb_groups; i++) {
        params[i].times.set_incubation(tinc);
        params[i].times.set_infectious_mild(tinfmild);
        params[i].times.set_serialinterval(tserint);
        params[i].times.set_hospitalized_to_home(thosp2home);
        params[i].times.set_home_to_hospitalized(thome2hosp);
        params[i].times.set_hospitalized_to_icu(thosp2icu);
        params[i].times.set_icu_to_home(ticu2home);
        params[i].times.set_infectious_asymp(tinfasy);
        params[i].times.set_icu_to_death(ticu2death);

        params[i].populations.set_total_t0(fact * nb_total_t0);
        params[i].populations.set_exposed_t0(fact * nb_exp_t0);
        params[i].populations.set_carrier_t0(fact * nb_car_t0);
        params[i].populations.set_infectious_t0(fact * nb_inf_t0);
        params[i].populations.set_hospital_t0(fact * nb_hosp_t0);
        params[i].populations.set_icu_t0(fact * nb_icu_t0);
        params[i].populations.set_recovered_t0(fact * nb_rec_t0);
        params[i].populations.set_dead_t0(fact * nb_dead_t0);

        params[i].probabilities.set_infection_from_contact(1.0);
        params[i].probabilities.set_asymp_per_infectious(alpha);
        params[i].probabilities.set_risk_from_symptomatic(beta);
        params[i].probabilities.set_hospitalized_per_infectious(rho);
        params[i].probabilities.set_icu_per_hospitalized(theta);
        params[i].probabilities.set_dead_per_icu(delta);
    }

    for (int i = 0; i < nb_groups; i++) {
        for (int j = i; j < nb_groups; j++) {
            contact_freq_matrix.set_cont_freq(fact * cont_freq, i, j);
        }
    }

    epi::ParameterSpace parameter_space(contact_freq_matrix, params, 0., 100., 0.2);

    std::vector<epi::SecirParams> params_sample = parameter_space.get_secir_params_sample();

    for (size_t i = 0; i < params_sample.size(); i++) {

        EXPECT_GE(params_sample[i].populations.get_total_t0(), 0);

        EXPECT_GE(params_sample[i].times.get_incubation_inv(), 0);

        EXPECT_GE(params_sample[i].probabilities.get_infection_from_contact(), 0);
    }

    epi::ContactFrequencyMatrix contact_sample = parameter_space.get_cont_freq_matrix_sample();

    for (size_t i = 0; i < params_sample.size(); i++) {
        for (size_t j = 0; j < params_sample.size(); j++) {
            EXPECT_GE(contact_sample.get_dampings(i, j).get_factor(1.0), 0);
            EXPECT_GE(contact_sample.get_cont_freq(i, j), 0);
        }
    }
}

TEST(ParameterStudies, test_normal_distribution)
{

    epi::ParameterDistributionNormal parameter_dist_normal_1;

    // check if standard deviation is reduced if between too narrow interval [min,max] has to be sampled.
    parameter_dist_normal_1.set_upper_bound(1);
    parameter_dist_normal_1.set_lower_bound(-1);
    parameter_dist_normal_1.set_log(false); // only avoid warning output in tests

    double std_dev_demanded = parameter_dist_normal_1.get_standard_dev();
    parameter_dist_normal_1.get_sample();

    EXPECT_GE(std_dev_demanded, parameter_dist_normal_1.get_standard_dev());

    // check if full argument constructor works correctly
    epi::ParameterDistributionNormal parameter_dist_normal_2(-1.0, 1.0, 0, parameter_dist_normal_1.get_standard_dev());

    EXPECT_EQ(parameter_dist_normal_1.get_lower_bound(), parameter_dist_normal_2.get_lower_bound());
    EXPECT_EQ(parameter_dist_normal_1.get_upper_bound(), parameter_dist_normal_2.get_upper_bound());
    EXPECT_EQ(parameter_dist_normal_1.get_mean(), parameter_dist_normal_2.get_mean());
    EXPECT_EQ(parameter_dist_normal_1.get_standard_dev(), parameter_dist_normal_2.get_standard_dev());

    // check if std_dev is not changed if boundaries are far enough away such that 99% of the density fits into the interval
    parameter_dist_normal_2.set_mean(5);
    parameter_dist_normal_2.set_standard_dev(1.5);
    parameter_dist_normal_2.set_lower_bound(1);
    parameter_dist_normal_2.set_upper_bound(10);
    std_dev_demanded = parameter_dist_normal_2.get_standard_dev();

    parameter_dist_normal_2.check_quantiles();
    EXPECT_EQ(std_dev_demanded, parameter_dist_normal_2.get_standard_dev());

    // check that sampling only occurs in boundaries
    int counter[10] = {0};
    for (int i = 0; i < 1000; i++) {
        double val = parameter_dist_normal_2.get_sample();
        EXPECT_GE(parameter_dist_normal_2.get_upper_bound() + 1e-10, val);
        EXPECT_LE(parameter_dist_normal_2.get_lower_bound() - 1e-10, val);
    }
}

TEST(ParameterStudies, test_uniform_distribution)
{

    // check if full argument constructor works correctly
    epi::ParameterDistributionUniform parameter_dist_unif(1.0, 10.0);

    EXPECT_EQ(parameter_dist_unif.get_lower_bound(), 1.0);
    EXPECT_EQ(parameter_dist_unif.get_upper_bound(), 10.0);

    // check that sampling only occurs in boundaries
    for (int i = 0; i < 1000; i++) {
        double val = parameter_dist_unif.get_sample();
        EXPECT_GE(parameter_dist_unif.get_upper_bound() + 1e-10, val);
        EXPECT_LE(parameter_dist_unif.get_lower_bound() - 1e-10, val);
    }
}

TEST(ParameterStudies, check_ensemble_run_result)
{
    double t0   = 0;
    double tmax = 50;
    double dt   = 0.1;

    double tinc    = 5.2, // R_2^(-1)+R_3^(-1)
        tinfmild   = 6, // 4-14  (=R4^(-1))
        tserint    = 4.2, // 4-4.4 // R_2^(-1)+0.5*R_3^(-1)
        thosp2home = 12, // 7-16 (=R5^(-1))
        thome2hosp = 5, // 2.5-7 (=R6^(-1))
        thosp2icu  = 2, // 1-3.5 (=R7^(-1))
        ticu2home  = 8, // 5-16 (=R8^(-1))
        tinfasy    = 6.2, // (=R9^(-1)=R_3^(-1)+0.5*R_4^(-1))
        ticu2death = 5; // 3.5-7 (=R5^(-1))

    double cont_freq = 0.5, // 0.2-0.75
        alpha        = 0.09, // 0.01-0.16
        beta         = 0.25, // 0.05-0.5
        delta        = 0.3, // 0.15-0.77
        rho          = 0.2, // 0.1-0.35
        theta        = 0.25; // 0.15-0.4

    double nb_total_t0 = 10000, nb_exp_t0 = 100, nb_inf_t0 = 50, nb_car_t0 = 50, nb_hosp_t0 = 20, nb_icu_t0 = 10,
           nb_rec_t0 = 10, nb_dead_t0 = 0;

    int nb_groups = 1;
    double fact   = 1.0 / (double)nb_groups;

    std::vector<epi::SecirParams> params{epi::SecirParams{}};
    epi::ContactFrequencyMatrix contact_freq_matrix{(size_t)nb_groups};
    for (size_t i = 1; i < nb_groups; i++) {
        params.push_back(epi::SecirParams{});
    }

    for (size_t i = 0; i < nb_groups; i++) {
        params[i].times.set_incubation(tinc);
        params[i].times.set_infectious_mild(tinfmild);
        params[i].times.set_serialinterval(tserint);
        params[i].times.set_hospitalized_to_home(thosp2home);
        params[i].times.set_home_to_hospitalized(thome2hosp);
        params[i].times.set_hospitalized_to_icu(thosp2icu);
        params[i].times.set_icu_to_home(ticu2home);
        params[i].times.set_infectious_asymp(tinfasy);
        params[i].times.set_icu_to_death(ticu2death);

        params[i].populations.set_total_t0(fact * nb_total_t0);
        params[i].populations.set_exposed_t0(fact * nb_exp_t0);
        params[i].populations.set_carrier_t0(fact * nb_car_t0);
        params[i].populations.set_infectious_t0(fact * nb_inf_t0);
        params[i].populations.set_hospital_t0(fact * nb_hosp_t0);
        params[i].populations.set_icu_t0(fact * nb_icu_t0);
        params[i].populations.set_recovered_t0(fact * nb_rec_t0);
        params[i].populations.set_dead_t0(fact * nb_dead_t0);

        params[i].probabilities.set_infection_from_contact(1.0);
        params[i].probabilities.set_asymp_per_infectious(alpha);
        params[i].probabilities.set_risk_from_symptomatic(beta);
        params[i].probabilities.set_hospitalized_per_infectious(rho);
        params[i].probabilities.set_icu_per_hospitalized(theta);
        params[i].probabilities.set_dead_per_icu(delta);
    }

    epi::Damping dummy(30., 0.3);
    for (int i = 0; i < nb_groups; i++) {
        for (int j = i; j < nb_groups; j++) {
            contact_freq_matrix.set_cont_freq(fact * cont_freq, i, j);
        }
    }

    epi::ParameterStudy parameter_study(
        [](double t0, double tmax, double dt, epi::ContactFrequencyMatrix const& contact_freq_matrix,
           std::vector<epi::SecirParams> const& params, std::vector<Eigen::VectorXd>& secir) {
            return epi::simulate(t0, tmax, dt, contact_freq_matrix, params, secir);
        },
        contact_freq_matrix, params, t0, tmax);

    // Run parameter study
    parameter_study.set_nb_runs(1);
    std::vector<std::vector<Eigen::VectorXd>> results = parameter_study.run();

    // printf("\n %d %d %d %d ", results.size(), results[0].size(), results[0][0].size(), params.size());
    for (size_t i = 0; i < results[0].size(); i++) {
        std::vector<double> total_at_ti(8, 0);
        // printf("\n");
        for (size_t j = 0; j < results[0][i].size(); j++) {
            // printf(" %.2e ( %d ) ", results[0][i][j], j % 8);
            total_at_ti[j % 8] += results[0][i][j];
        }
        for (size_t j = 0; j < params.size(); j++) {
            // EXPECT_NEAR(total_at_ti[j], params[j].populations.get_total_t0(), 1e-3) << " day " << i << " group " << j;
        }
    }
}