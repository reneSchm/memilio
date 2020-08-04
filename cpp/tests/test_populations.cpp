#include "epidemiology/populations.h"
#include <gtest/gtest.h>

enum Category
{
    InfectionTypeCategory,
    AgeGroupCategory,
    ContinentCategory
};

enum InfectionType
{
    S,
    E,
    C,
    I,
    H,
    U,
    R,
    D,
    InfectionTypeCount
};

enum AgeGroup
{
    TwelveAndYounger,
    TwelveToTwenty,
    TwentyToThirtyfive,
    ThirtyfiveToFourtyFive,
    FourtyFiveToSixtyFive,
    SixtyFiveToSeventyFive,
    SeventyFiveAndOlder,
    AgeGroupCount
};

enum Continent
{
    Europe,
    Asia,
    NorthAmerica,
    SouthAmerica,
    Australia,
    Antarctica,
    Africa,
    ContinentCount
};

TEST(TestPopulations, sizes)
{

    int num_compartments = InfectionTypeCount * AgeGroupCount * ContinentCount;
    ASSERT_EQ(7 * 7 * 8, num_compartments);

    epi::Populations m({InfectionTypeCount, AgeGroupCount, ContinentCount});

    ASSERT_EQ(num_compartments, m.get_num_compartments());
    ASSERT_EQ(num_compartments, m.get_compartments().size());
    auto category_sizes = m.get_category_sizes();
    ASSERT_EQ(category_sizes[0], InfectionTypeCount);
    ASSERT_EQ(category_sizes[1], AgeGroupCount);
    ASSERT_EQ(category_sizes[2], ContinentCount);
    ASSERT_EQ(0, m.get_total());
}

TEST(TestPopulations, set_population)
{
    epi::Populations m({InfectionTypeCount, AgeGroupCount, ContinentCount});
    m.set_total(1.);
    int num_compartments = InfectionTypeCount * AgeGroupCount * ContinentCount;

    for (size_t i = 0; i < InfectionTypeCount; ++i) {
        for (size_t j = 0; j < AgeGroupCount; ++j) {
            for (size_t k = 0; k < ContinentCount; ++k) {
                ASSERT_NEAR(1. / num_compartments, m.get({i, j, k}), 1e-12);
            }
        }
    }
    ASSERT_NEAR(1., m.get_total(), 1e-12);
}

TEST(TestPopulations, group_population)
{
    epi::Populations m({InfectionTypeCount, AgeGroupCount, ContinentCount});
    m.set_total(1.);
    int num_compartments = InfectionTypeCount * AgeGroupCount * ContinentCount;

    ASSERT_NEAR(1. / AgeGroupCount, m.get_group_total(AgeGroupCategory, FourtyFiveToSixtyFive), 1e-12);
    m.set_group_total(AgeGroupCategory, FourtyFiveToSixtyFive, 1.);
    ASSERT_NEAR(1., m.get_group_total(AgeGroupCategory, FourtyFiveToSixtyFive), 1e-12);
    ASSERT_NEAR(2 - 1. / AgeGroupCount, m.get_total(), 1e-12);

    Eigen::VectorXd y = m.get_compartments();
    size_t idx        = 0;
    for (size_t i = 0; i < InfectionTypeCount; ++i) {
        for (size_t j = 0; j < AgeGroupCount; ++j) {
            for (size_t k = 0; k < ContinentCount; ++k) {
                ASSERT_EQ(idx, m.get_flat_index({i, j, k}));

                if (j == FourtyFiveToSixtyFive) {
                    ASSERT_NEAR(y[idx], 1. / (InfectionTypeCount * ContinentCount), 1e-12);
                }
                else {
                    ASSERT_NEAR(y[idx], 1. / num_compartments, 1e-12);
                }
                idx++;
            }
        }
    }
}

TEST(TestPopulations, set_difference_from_total)
{
    epi::Populations m({InfectionTypeCount, AgeGroupCount, ContinentCount});
    m.set({S, TwentyToThirtyfive, Africa}, 100);

    m.set_difference_from_total({E, TwentyToThirtyfive, Africa}, 1000);
    ASSERT_NEAR(1000, m.get_total(), 1e-12);
    ASSERT_NEAR(900, m.get({E, TwentyToThirtyfive, Africa}), 1e-12);

    m.set_difference_from_total({E, TwentyToThirtyfive, Africa}, 2000);
    ASSERT_NEAR(2000, m.get_total(), 1e-12);
    ASSERT_NEAR(1900, m.get({E, TwentyToThirtyfive, Africa}), 1e-12);

    for (size_t i = 0; i < InfectionTypeCount; ++i) {
        for (size_t j = 0; j < AgeGroupCount; ++j) {
            for (size_t k = 0; k < ContinentCount; ++k) {
                if (i == S && j == TwentyToThirtyfive && k == Africa) {
                    ASSERT_NEAR(100, m.get({i, j, k}), 1e-12);
                }
                else if (i == E && j == TwentyToThirtyfive && k == Africa) {
                    ASSERT_NEAR(1900, m.get({i, j, k}), 1e-12);
                }
                else {
                    ASSERT_NEAR(0, m.get({i, j, k}), 1e-12);
                }
            }
        }
    }
}

TEST(TestPopulations, set_difference_from_group_total)
{
    epi::Populations m({InfectionTypeCount, AgeGroupCount, ContinentCount});
    m.set({S, TwentyToThirtyfive, Africa}, 100);
    m.set({S, TwentyToThirtyfive, Europe}, 200);

    m.set_difference_from_group_total({E, TwentyToThirtyfive, Africa}, ContinentCategory, Africa, 1000);
    ASSERT_NEAR(1000, m.get_group_total(ContinentCategory, Africa), 1e-12);
    ASSERT_NEAR(900, m.get({E, TwentyToThirtyfive, Africa}), 1e-12);
    ASSERT_NEAR(1200, m.get_total(), 1e-12);

    m.set_difference_from_group_total({E, TwentyToThirtyfive, Africa}, ContinentCategory, Africa, 2000);
    ASSERT_NEAR(2000, m.get_group_total(ContinentCategory, Africa), 1e-12);
    ASSERT_NEAR(1900, m.get({E, TwentyToThirtyfive, Africa}), 1e-12);
    ASSERT_NEAR(2200, m.get_total(), 1e-12);

    for (size_t i = 0; i < InfectionTypeCount; ++i) {
        for (size_t j = 0; j < AgeGroupCount; ++j) {
            for (size_t k = 0; k < ContinentCount; ++k) {
                if (i == S && j == TwentyToThirtyfive && k == Africa) {
                    ASSERT_NEAR(100, m.get({i, j, k}), 1e-12);
                }
                else if (i == E && j == TwentyToThirtyfive && k == Africa) {
                    ASSERT_NEAR(1900, m.get({i, j, k}), 1e-12);
                }
                else if (i == S && j == TwentyToThirtyfive && k == Europe) {
                    ASSERT_NEAR(200, m.get({i, j, k}), 1e-12);
                }
                else {
                    ASSERT_NEAR(0, m.get({i, j, k}), 1e-12);
                }
            }
        }
    }
}
