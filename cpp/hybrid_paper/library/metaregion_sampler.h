#ifndef METAREGION_SAMPLER_H_
#define METAREGION_SAMPLER_H_

#include "memilio/math/eigen.h"
#include "memilio/utils/random_number_generator.h"

#include <vector>

class MetaregionSampler
{
public:
    MetaregionSampler(Eigen::Ref<const Eigen::MatrixXi> metaregions)
        : m_metaregion_positions(metaregions.maxCoeff())
    {
        Eigen::MatrixXi is_outside = (1 - metaregions.cast<bool>().array()).matrix().cast<int>();
        extend_bitmap(is_outside, 8);
        for (Eigen::Index i = 0; i < metaregions.rows(); i++) {
            for (Eigen::Index j = 0; j < metaregions.cols(); j++) {
                if (metaregions(i, j) > 0 && !is_outside(i, j)) {
                    m_metaregion_positions[metaregions(i, j) - 1].emplace_back(i, j);
                }
            }
        }
    }

    void extend_bitmap(Eigen::Ref<Eigen::MatrixXi> bitmap, Eigen::Index width)
    {
        Eigen::Index extent    = 2 * width + 1;
        Eigen::MatrixXi canvas = Eigen::MatrixXi::Zero(bitmap.rows() + extent, bitmap.cols() + extent);
        for (Eigen::Index i = 0; i < bitmap.rows(); i++) {
            for (Eigen::Index j = 0; j < bitmap.cols(); j++) {
                for (Eigen::Index k = 0; k < extent; k++) {
                    for (Eigen::Index l = 0; l < extent; l++) {
                        canvas(i + k, j + l) |= bitmap(i, j);
                    }
                }
            }
        }
        bitmap = canvas.block(width, width, bitmap.rows(), bitmap.cols());
    }

    Eigen::Vector2d operator()(size_t metaregion_index) const
    {
        // TODO: discuss implementation: Which positions should be eligible? What distribution do we choose?
        // Suggestion: pick only points from within the circle of maximum radius fitting into a region.
        size_t position_index = mio::DiscreteDistribution<int>::get_instance()(
            std::vector<double>(m_metaregion_positions[metaregion_index].size(), 1));
        return m_metaregion_positions[metaregion_index][position_index];
    }

private:
    std::vector<std::vector<Eigen::Vector2d>> m_metaregion_positions;
};

#endif //METAREGION_SAMPLER_H_