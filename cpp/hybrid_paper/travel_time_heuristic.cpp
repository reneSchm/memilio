#include "memilio/utils/parameter_distributions.h"
#include "mpm/potentials/commuting_potential.h"
#include "memilio/math/eigen.h"

inline double square(double x)
{
    return x * x;
}

template <class Vec>
double mean(Vec v)
{
    return v.sum() / v.size();
}

template <class Vec>
double variance(Vec v)
{
    return mean(v.array().square()) - square(mean(v));
}

int main()
{
    double t  = 0;
    double dt = 0.1;

    constexpr unsigned n = 10000000;

    Eigen::VectorXd t_return(n), t_depart(n);

    auto return_dist = mio::ParameterDistributionNormal(9.0 / 24.0, 23.0 / 24.0, 18.0 / 24.0);

    for (int i = 0; i < n; i++) {
        t_return[i] = t + return_dist.get_rand_sample();
        t_depart[i] = TriangularDistribution(t_return[i] - dt, t, t + 9.0 / 24.0).get_instance();
    }

    auto t_travel = t_return - t_depart;

    std::cout << "number of samples: " << n << "\n\n";
    std::cout << "normal dist min:  " << t_return.minCoeff() << "      \t expected: " << return_dist.get_lower_bound()
              << "\n"
              << "normal dist max:  " << t_return.maxCoeff() << "      \t expected: " << return_dist.get_upper_bound()
              << "\n"
              << "normal dist mean: " << mean(t_return) << "      \t expected: " << return_dist.get_mean() << "\n"
              << "normal dist var:  " << variance(t_return)
              << "      \t expected: " << square(return_dist.get_standard_dev()) << "\n"
              << "\n";
    std::cout << "travel time min:  " << t_travel.minCoeff() << "      \t in hours: " << t_travel.minCoeff() * 24
              << "\n"
              << "travel time max:  " << t_travel.maxCoeff() << "      \t in hours: " << t_travel.maxCoeff() * 24
              << "\n"
              << "travel time mean: " << mean(t_travel) << "      \t in hours: " << mean(t_travel) * 24 << "\n"
              << "travel time var:  " << variance(t_travel) << "      \t in hours: " << variance(t_travel) * 24 << "\n";
    return 0;
}