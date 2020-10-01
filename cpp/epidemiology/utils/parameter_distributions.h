#ifndef PARAMETER_DISTRIBUTIONS_H
#define PARAMETER_DISTRIBUTIONS_H

#include "epidemiology/utils/logging.h"
#include "epidemiology/utils/visitor.h"

#include <vector>
#include <random>

namespace epi
{

/* TODO: Add more distributions here. */
typedef enum
{
    DIST_UNIFORM,
    DIST_NORMAL
} distribution;

/**
 * @brief This is a visitor class to visit all Parameter Distribution objects
 *
 * More information to the visitor pattern is here: https://en.wikipedia.org/wiki/Visitor_pattern
 */
using ParameterDistributionVisitor = Visitor<class ParameterDistributionNormal, class ParameterDistributionUniform>;
using ConstParameterDistributionVisitor =
    ConstVisitor<class ParameterDistributionNormal, class ParameterDistributionUniform>;

template <class Derived>
using VisitableParameterDistribution =
    Visitable<Derived, class ParameterDistribution, ParameterDistributionVisitor, ConstParameterDistributionVisitor>;

/*
 * Parameter Distribution class which contains the name of a variable as string
 * the lower bound and the upper bound as maximum admissible values and an enum
 * item with the name of the distribution
 */
class ParameterDistribution
{
public:
    ParameterDistribution()
        : m_lower_bound{0}
        , m_upper_bound{0}
    {
        std::random_device random_device;
        m_random_generator = std::mt19937{random_device()};
    }

    ParameterDistribution(double lower_bound, double upper_bound)
    {
        std::random_device random_device;
        m_random_generator = std::mt19937{random_device()};
        m_lower_bound      = lower_bound;
        m_upper_bound      = upper_bound;
    }

    virtual ~ParameterDistribution() = default;

    void set_lower_bound(double lower_bound)
    {
        m_lower_bound = lower_bound;
    }

    void set_upper_bound(double upper_bound)
    {
        m_upper_bound = upper_bound;
    }

    void add_predefined_sample(double sample)
    {
        m_predefined_samples.push_back(sample);
    }

    void remove_predefined_samples()
    {
        m_predefined_samples.resize(0);
    }

    const std::vector<double>& get_predefined_samples() const
    {
        return m_predefined_samples;
    }

    double get_lower_bound() const
    {
        return m_lower_bound;
    }

    double get_upper_bound() const
    {
        return m_upper_bound;
    }

    /*
     * @brief returns a value for the given parameter distribution
     * in case some predefined samples are set, these values are taken
     * first, in case the vector of predefined values is empty, a 'real'
     * random sample is taken
     */
    double get_sample()
    {
        if (m_predefined_samples.size() > 0) {
            double rnumb = m_predefined_samples[0];
            m_predefined_samples.erase(m_predefined_samples.begin());
            return rnumb;
        }
        else {
            return get_rand_sample();
        }
    }

    virtual double get_rand_sample() = 0;

    virtual ParameterDistribution* clone() const = 0;

    /**
     * @brief This function implements the visit interface of the visitor pattern
     *
     * It can be used for any ways of working with the class to dispatch
     * the class type. More information here: https://en.wikipedia.org/wiki/Visitor_pattern
     */
    virtual void accept(ParameterDistributionVisitor& visitor)            = 0;
    virtual void accept(ConstParameterDistributionVisitor& visitor) const = 0;

protected:
    double m_lower_bound; /*< A realistic lower bound on the given parameter */
    double m_upper_bound; /*< A realistic upper bound on the given parameter */
    std::mt19937 m_random_generator;
    std::vector<double>
        m_predefined_samples; // if these values are set; no real sample will occur but these values will be taken
};

/*
 * Child class of Parameter Distribution class which additionally contains
 * the mean value and the standard deviation for a normal distribution
 */
class ParameterDistributionNormal : public VisitableParameterDistribution<ParameterDistributionNormal>
{
public:
    ParameterDistributionNormal()
        : VisitableParameterDistribution<ParameterDistributionNormal>()
    {
        m_mean         = 0;
        m_standard_dev = 1;
    }

    ParameterDistributionNormal(double mean, double standard_dev)
        : VisitableParameterDistribution<ParameterDistributionNormal>()
    {
        m_mean         = mean;
        m_standard_dev = standard_dev;
        check_quantiles(m_mean, m_standard_dev);
    }

    ParameterDistributionNormal(double lower_bound, double upper_bound, double mean)
        : VisitableParameterDistribution<ParameterDistributionNormal>(lower_bound, upper_bound)
    {
        m_mean         = mean;
        m_standard_dev = upper_bound; // set as to high and adapt then
        adapt_standard_dev(m_standard_dev);
    }

    ParameterDistributionNormal(double lower_bound, double upper_bound, double mean, double standard_dev)
        : VisitableParameterDistribution<ParameterDistributionNormal>(lower_bound, upper_bound)
    {
        m_mean         = mean;
        m_standard_dev = standard_dev;
        check_quantiles(m_mean, m_standard_dev);
    }

    void set_mean(double mean)
    {
        m_mean = mean;
    }

    bool check_quantiles()
    {
        return check_quantiles(m_mean, m_standard_dev);
    }

    /*
     * @brief verification that at least 99% of the density
     * function lie in the interval defined by the boundaries
     */
    bool check_quantiles(double& mean, double& standard_dev)
    {
        bool changed = false;

        changed = adapt_mean(mean);

        changed = adapt_standard_dev(standard_dev);

        if (changed && m_log_stddev_change) {
            log_warning("Standard deviation reduced to fit 99% of the distribution within [lowerbound,upperbound].");
        }

        return changed;
    }

    bool adapt_mean(double& mean)
    {
        bool changed = false;
        if (mean < m_lower_bound || mean > m_upper_bound) {
            mean    = 0.5 * (m_upper_bound - m_lower_bound);
            changed = true;
        }
        return changed;
    }

    // ensure that 0.99 % of the distribution are within lower bound and upper bound
    bool adapt_standard_dev(double& standard_dev)
    {
        bool changed = false;
        if (m_mean + standard_dev * m_quantile > m_upper_bound) {
            standard_dev = (m_upper_bound - m_mean) / m_quantile;
            changed      = true;
        }
        if (m_mean - standard_dev * m_quantile < m_lower_bound) {
            standard_dev = (m_mean - m_lower_bound) / m_quantile;
            changed      = true;
        }

        return changed;
    }

    void set_standard_dev(double standard_dev)
    {
        m_standard_dev = standard_dev;
    }

    double get_mean() const
    {
        return m_mean;
    }

    double get_standard_dev() const
    {
        return m_standard_dev;
    }

    void log_stddev_changes(bool log_stddev_change)
    {
        m_log_stddev_change = log_stddev_change;
    }

    /*
     * @brief gets a sample of a normally distributed variable
     * before sampling, it is verified that at least 99% of the
     * density function lie in the interval defined by the boundaries
     * otherwise the normal distribution is adapted
     */
    double get_rand_sample() override
    {
        if (check_quantiles(m_mean, m_standard_dev) || m_distribution.mean() != m_mean ||
            m_distribution.stddev() != m_standard_dev) {
            m_distribution = std::normal_distribution<double>{m_mean, m_standard_dev};
        }

        int i        = 0;
        int retries  = 10;
        double rnumb = m_distribution(m_random_generator);
        while ((rnumb > m_upper_bound || rnumb < m_lower_bound) && i < retries) {
            rnumb = m_distribution(m_random_generator);
            i++;
            if (i == retries) {
                log_warning("Not successfully sampled within [min,max].");
                if (rnumb > m_upper_bound) {
                    rnumb = m_upper_bound;
                }
                else {
                    rnumb = m_lower_bound;
                }
            }
        }
        return rnumb;
    }

    ParameterDistribution* clone() const override
    {
        return new ParameterDistributionNormal(*this);
    }

private:
    double m_mean; // the mean value of the normal distribution
    double m_standard_dev; // the standard deviation of the normal distribution
    constexpr static double m_quantile = 2.5758; // 0.995 quartile
    std::normal_distribution<double> m_distribution;
    bool m_log_stddev_change = true;
};

/*
 * Child class of Parameter Distribution class which represents an uniform distribution 
 */
class ParameterDistributionUniform : public VisitableParameterDistribution<ParameterDistributionUniform>
{
public:
    ParameterDistributionUniform()
        : VisitableParameterDistribution<ParameterDistributionUniform>()
    {
    }

    ParameterDistributionUniform(double lower_bound, double upper_bound)
        : VisitableParameterDistribution<ParameterDistributionUniform>(lower_bound, upper_bound)
    {
    }

    /*
     * @brief gets a sample of a uniformly distributed variable
     */
    double get_rand_sample() override
    {
        if (m_distribution.max() != m_upper_bound || m_distribution.min() != m_lower_bound) {
            m_distribution = std::uniform_real_distribution<double>{m_lower_bound, m_upper_bound};
        }

        return m_distribution(m_random_generator);
    }

    ParameterDistribution* clone() const override
    {
        return new ParameterDistributionUniform(*this);
    }

private:
    std::uniform_real_distribution<double> m_distribution;
};

} // namespace epi

#endif // PARAMETER_DISTRIBUTIONS_H