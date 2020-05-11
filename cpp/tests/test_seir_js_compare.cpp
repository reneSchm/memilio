#include "load_test_data.h"
#include <gtest/gtest.h>
#include <epidemiology/seir.h>

using real = double;

class TestCompareSeirWithJS : public testing::Test
{
protected:
    void SetUp() override
    {
        refData = load_test_data_csv<real>("data/seir-js-compare.csv");
        t0      = 0.;
        tmax    = 50.;
        dt      = 1.002003997564315796e-01;

        params.nb_exp_t0    = 10000;
        params.nb_inf_t0    = 1000;
        params.nb_total_t0  = 1061000;
        params.nb_rec_t0    = 1000;
        params.nb_sus_t0    = params.nb_total_t0 - params.nb_exp_t0 - params.nb_inf_t0 - params.nb_rec_t0;
        params.tinc_inv     = 1. / 5.2;
        params.cont_freq    = 2.7;
        params.tinfmild_inv = 0.5;

        // add two dampings
        params.dampings.add(epi::Damping(0., 1.0));
        params.dampings.add(epi::Damping(12., 0.4));
    }

public:
    std::vector<std::vector<real>> refData;
    real t0;
    real tmax;
    real dt;
    epi::SeirParams params;
};

TEST_F(TestCompareSeirWithJS, integrate)
{
    EXPECT_EQ(500, refData.size());

    std::vector<std::vector<real>> result(0);

    simulate(t0, tmax, dt, params, result);

    EXPECT_EQ(500, result.size());

    for (size_t irow = 0; irow < 500; ++irow) {
        double t = refData[irow][0];
        for (size_t icol = 0; icol < 4; ++icol) {
            double ref    = refData[irow][icol + 1];
            double actual = result[irow][icol];

            double tol = 1e-6 * ref;
            EXPECT_NEAR(ref, actual, tol);
        }
    }
}