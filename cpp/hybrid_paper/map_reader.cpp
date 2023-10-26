// WORK IN PROGRESS - use at your own risk
// compile command (requires a proir successfull build of memilio):
// g++ --std=c++14 -Wall --pedantic -Icpp -Icpp/build/_deps/eigen-src -o map_reader map_reader.cpp cpp/models/mpm/potentials/map_reader.cpp
// generate pgm's using map.py with .shp files from https://daten.gdz.bkg.bund.de/produkte/vg/vg2500/aktuell/vg2500_12-31.gk3.shape.zip (unpack into tools/)

#include "models/mpm/potentials/map_reader.h"
#include "memilio/utils/logging.h"
#include "boost/filesystem.hpp"

using namespace mio::mpm;

typedef Eigen::Vector2d Position;
typedef double ScalarType;
namespace fs = boost::filesystem;

// Add neighbouring (matrix) indices to queue
// void enqueue_neighbours_if(const std::pair<ScalarType, ScalarType>& index, const std::pair<ScalarType, ScalarType>& max,
//                            std::deque<std::pair<size_t, size_t>>& queue,
//                            const std::function<bool(size_t, size_t)>& predicate)
// {
//     if (index.first > 0 && predicate(index.first - 1, index.second)) {
//         queue.push_back({index.first - 1, index.second});
//     }
//     if (index.first + 1 < max.first && predicate(index.first + 1, index.second)) {
//         queue.push_back({index.first + 1, index.second});
//     }
//     if (index.second > 0 && predicate(index.first, index.second - 1)) {
//         queue.push_back({index.first, index.second - 1});
//     }
//     if (index.second + 1 < max.second && predicate(index.first, index.second + 1)) {
//         queue.push_back({index.first, index.second + 1});
//     }
// }

// Add neighbouring (matrix) indices to queue
void enqueue_neighbours(const std::pair<ScalarType, ScalarType>& index, const std::pair<ScalarType, ScalarType>& max,
                        Eigen::Ref<Eigen::MatrixXi> mask, std::deque<std::pair<size_t, size_t>>& queue)
{
    // mask encoding : -1 - not , 1 - identified
    if (index.first > 0 && mask(index.first - 1, index.second) == 0) {
        mask(index.first - 1, index.second) = -1;
        queue.push_back({index.first - 1, index.second});
    }
    if (index.first + 1 < max.first && mask(index.first + 1, index.second) == 0) {
        mask(index.first + 1, index.second) = -1;
        queue.push_back({index.first + 1, index.second});
    }
    if (index.second > 0 && mask(index.first, index.second - 1) == 0) {
        mask(index.first, index.second - 1) = -1;
        queue.push_back({index.first, index.second - 1});
    }
    if (index.second + 1 < max.second && mask(index.first, index.second + 1) == 0) {
        mask(index.first, index.second + 1) = -1;
        queue.push_back({index.first, index.second + 1});
    }
}

/**
 * @brief Find a connected region around a starting point in an image of similar color.
 * @param image A matrix interpreted as greyscale image.
 * @param start_x Column index for image inside a connected region.
 * @param start_y Row index for image inside a connected region.
 * @param color_tolerance Absolute tolerance for matching adjacent colors.
 * @return A matrix with entries 1.0 if the point is within the connected region, 0 otherwise.
 */
Eigen::MatrixXi find_connected_image_region(Eigen::Ref<const Eigen::MatrixXd> image, const size_t start_x,
                                            const size_t start_y, const ScalarType color_tolerance = 0)
{
    // setup
    size_t x, y;
    ScalarType color = image(start_x, start_y);
    std::deque<std::pair<size_t, size_t>> queue({{start_x, start_y}});
    Eigen::MatrixXi mask = Eigen::MatrixXi::Zero(image.rows(), image.cols());
    // iterate (BFS)
    for (; not queue.empty(); queue.pop_front()) {
        std::tie(x, y) = queue.front();
        if (std::abs(color - image(x, y)) <= color_tolerance) {
            mask(x, y) = 1;
            enqueue_neighbours({x, y}, {image.rows(), image.cols()}, mask, queue);
        }
        else {
            mask(x, y) = 0;
        }
    }
    return mask;
}

// Eigen::MatrixXd find_bounded_image_region(Eigen::Ref<Eigen::MatrixXd> image, const size_t start_x, const size_t start_y,
//                                           ScalarType boundary_color, ScalarType color_tolerance = 0);

// apply a weight stencil to each entry of image, merging the results using the elementwise maximum
// should work with any stencil, but odd number of rows/cols is recommended
void apply_stencil(Eigen::Ref<Eigen::MatrixXd> image, Eigen::Ref<const Eigen::MatrixXd> stencil, ScalarType color,
                   ScalarType color_tolerance = 0)
{
    const Eigen::Index rows = stencil.rows();
    const Eigen::Index cols = stencil.cols();
    assert(rows > 0 && cols > 0);
    Eigen::MatrixXd canvas = Eigen::MatrixXd::Zero(image.rows() + rows, image.cols() + cols);
    // canvas.block(rows / 2, cols / 2, image.rows(), image.cols()) = image;
    for (Eigen::Index i = 0; i < image.rows(); i++) {
        for (Eigen::Index j = 0; j < image.cols(); j++) {
            if (std::abs(image(i, j) - color) <= color_tolerance) {
                canvas.block(i, j, rows, cols) =
                    canvas.block(i, j, rows, cols).array().max((image(i, j) * stencil).array());
            }
        }
    }
    image = canvas.block(rows / 2, cols / 2, image.rows(), image.cols());
}

// extend a bitmap in each direction by width, merging entries using bitwise or
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

// metaregions are now  detected dynamically. the county data is unused
// std::vector<Position> laender = {
//     {0.16458333333333333, 0.4479166666666667}, // Schleswig-Holstein
//     {0.21666666666666667, 0.6333333333333333}, // MVP
//     {0.23958333333333334, 0.46458333333333335}, // Hamburg
//     {0.28958333333333336, 0.38958333333333334}, // Bremen
//     {0.3020833333333333, 0.4479166666666667}, // Niedersachsen
//     {0.30416666666666664, 0.6916666666666667}, // Brandenburg
//     {0.34375, 0.69375}, // Berlin
//     {0.4270833333333333, 0.58125}, // Sachsen Anhalt
//     {0.46875, 0.2916666666666667}, // NRW
//     {0.5083333333333333, 0.6979166666666666}, // Sachsen
//     {0.525, 0.5354166666666667}, // Thüringen
//     {0.5895833333333333, 0.38958333333333334}, // Hessen
//     {0.6354166666666666, 0.28541666666666665}, // RLP
//     {0.69375, 0.25416666666666665}, // Saarland
//     {0.74375, 0.6104166666666667}, // Bayern
//     {0.8, 0.4}, // BaWü
// };

int main()
{
    const std::string file_prefix                = "../../";
    const std::string potential_filename         = file_prefix + "potential_dpi=300.pgm";
    const std::string output_potential           = file_prefix + "potentially_germany.pgm";
    const std::string output_metaregions         = file_prefix + "metagermany.pgm";
    const std::string output_boundary_ids        = file_prefix + "boundary_ids.pgm";
    const std::string output_boundary_simplyfied = file_prefix + "boundary_simplyfied.pgm";

    mio::log_info("open file");
    std::ifstream file(potential_filename);
    if (!file.is_open()) { // write error and abort
        mio::log(mio::LogLevel::critical, "Could not open file {}", potential_filename);
        return 1;
    }

    mio::log_info("read file");
    auto image = read_pgm(file);
    file.close();
    image = (1 - image.array()).matrix(); // invert colors

    //mio::log_info("manually fix an isolated pixel in the corner Fuerstenfeldbruck/LH Muenchen/Muenchen");
    //image(508, 574) = 1;

    mio::log_info("map metaregions");
    size_t region              = 1;
    Eigen::MatrixXi is_outside = find_connected_image_region(image, 0, 0);
    extend_bitmap(is_outside, 1);
    Eigen::MatrixXi metaregions = Eigen::MatrixXi::Zero(image.rows(), image.cols());
    for (Eigen::Index i = 0; i < image.rows(); i++) {
        for (Eigen::Index j = 0; j < image.cols(); j++) {
            if (!is_outside(i, j) && image(i, j) == 0 && metaregions(i, j) == 0) {
                auto mask = find_connected_image_region(image, i, j);
                if(mask.array().sum()<500) continue;
                metaregions += region * mask;
                region++;
            }
        }
    }
    mio::log_info(" -> found {}", region - 1);

    mio::log_info("set stencil");
    // clang-format off
    // generated using https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma/55737551#55737551
    // with: N = 9, Sigma = [[1,0],[0,1]]
    // stored: np.savetxt("tst.txt", (Z - Z.min())/(Z.max() - Z.min()))
    // used only central 7x7 data
    std::vector<ScalarType> gauss_data = {
        8.870833551280334073e-02, 1.819281669245390864e-01, 2.731928597373864953e-01, 3.120522650710688128e-01, 2.731928597373864953e-01, 1.819281669245390864e-01, 8.870833551280334073e-02,
        1.819281669245390864e-01, 3.560857401120277599e-01, 5.265906335159235008e-01, 5.991895604387955654e-01, 5.265906335159235008e-01, 3.560857401120277599e-01, 1.819281669245390864e-01,
        2.731928597373864953e-01, 5.265906335159235008e-01, 7.746737895689834730e-01, 8.803046049522565974e-01, 7.746737895689834730e-01, 5.265906335159235008e-01, 2.731928597373864953e-01,
        3.120522650710688128e-01, 5.991895604387955654e-01, 8.803046049522565974e-01, 1.000000000000000000e+00, 8.803046049522565974e-01, 5.991895604387955654e-01, 3.120522650710688128e-01,
        2.731928597373864953e-01, 5.265906335159235008e-01, 7.746737895689834730e-01, 8.803046049522565974e-01, 7.746737895689834730e-01, 5.265906335159235008e-01, 2.731928597373864953e-01,
        1.819281669245390864e-01, 3.560857401120277599e-01, 5.265906335159235008e-01, 5.991895604387955654e-01, 5.265906335159235008e-01, 3.560857401120277599e-01, 1.819281669245390864e-01,
        8.870833551280334073e-02, 1.819281669245390864e-01, 2.731928597373864953e-01, 3.120522650710688128e-01, 2.731928597373864953e-01, 1.819281669245390864e-01, 8.870833551280334073e-02
    };
    // clang-format on
    Eigen::Matrix<double, 7, 7> stencil;
    for (int i = 0; i < 7; i++)
        for (int j = 0; j < 7; j++)
            stencil(i, j) = gauss_data[i + 7 * j];

    Eigen::Matrix<double, 1, 5> stencil_ext;
    stencil_ext(0, 0) = 0.5;
    stencil_ext(0, 1) = 0.75;
    stencil_ext(0, 2) = 1.0;
    stencil_ext(0, 3) = 0.75;
    stencil_ext(0, 4) = 0.5;

    mio::log_info("identify boundary segments");
    Eigen::MatrixXi boundaries            = Eigen::MatrixXi::Zero(image.rows(), image.cols());
    Eigen::MatrixXi boundaries_simplified = Eigen::MatrixXi::Zero(image.rows(), image.cols());
    // iterate image, leave out two outermost rows/cols
    int check_width = 3;
    for (Eigen::Index i = check_width; i < image.rows() - check_width; i++) {
        for (Eigen::Index j = check_width; j < image.cols() - check_width; j++) {
            // skip interior
            if (metaregions(i, j) != 0 or is_outside(i, j))
                continue;
            // look for land ids in a 5x5 square
            u_int16_t ids = 0;
            for (int k = -check_width; k <= check_width; k++) {
                for (int l = -check_width; l <= check_width; l++) {
                    ids |= 1 << (int)(metaregions(i + k, j + l) - 1);
                }
            }
            if (num_bits_set(ids) > 1) { // 0 or 1 should only occur on exterior pixels
                boundaries(i, j)            = ids;
                boundaries_simplified(i, j) = 2;
            }
            else {
                boundaries_simplified(i, j) = 1;
            }
            assert(num_bits_set(ids) < 4);
        }
    }
    extend_bitmap(boundaries, stencil.cols() / 2);
    extend_bitmap(boundaries_simplified, stencil.cols() / 2);

    mio::log_info("apply stencil");
    apply_stencil(image, stencil, 1.0);

    mio::log_info("expand border");
    Eigen::MatrixXd exterior = find_connected_image_region(image, 0, 0, 0.75).cast<ScalarType>();
    apply_stencil(exterior, stencil_ext.transpose() * stencil_ext, 1.0);
    image = image.array().max(4 * exterior.array()).matrix();
    // image = 4 * exterior;

    mio::log_info("write files");
    std::ofstream ofile(output_potential);
    if (!ofile.is_open()) {
        mio::log(mio::LogLevel::critical, "Could not open file {}", output_potential);
        return 1;
    }
    write_pgm(ofile, image);
    ofile.close();
    ofile.open(output_metaregions);
    if (!ofile.is_open()) {
        mio::log(mio::LogLevel::critical, "Could not open file {}", output_metaregions);
        return 1;
    }
    write_pgm(ofile, metaregions);
    ofile.close();
    ofile.open(output_boundary_ids);
    if (!ofile.is_open()) {
        mio::log(mio::LogLevel::critical, "Could not open file {}", output_boundary_ids);

        return 1;
    }
    write_pgm(ofile, boundaries);
    ofile.close();
    ofile.open(output_boundary_simplyfied);
    if (!ofile.is_open()) {
        mio::log(mio::LogLevel::critical, "Could not open file {}", output_boundary_simplyfied);
        return 1;
    }
    write_pgm(ofile, boundaries_simplified);
    ofile.close();

    mio::log_info("success");
}
