// WORK IN PROGRESS - use at your own risk
// compile command (requires a proir successfull build of memilio):
// g++ --std=c++14 -Wall --pedantic -o map_reader map_reader.cpp
// generate pgm's using map.py with .shp files from https://daten.gdz.bkg.bund.de/produkte/vg/vg2500/aktuell/vg2500_12-31.gk3.shape.zip (unpack into tools/)

#include <deque>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#include "build/_deps/eigen-src/Eigen/Core"
#define DEBUG(cout_args) std::cerr << cout_args << std::endl << std::flush;
typedef double ScalarType;

Eigen::MatrixXd read_pgm(std::istream& pgm_file)
{
    size_t height, width, color_range;
    std::string reader;
    std::stringstream parser;
    // ignore magic number (P2)
    std::getline(pgm_file, reader);
    // get dims
    std::getline(pgm_file, reader);
    parser.str(reader);
    parser >> width >> height;
    // get color range (max value for colors)
    std::getline(pgm_file, reader);
    parser.clear();
    parser.str(reader);
    parser >> color_range;
    // read image data
    // we assume (0,0) to be at the bottom left
    Eigen::MatrixXd data(width, height);
    for (size_t j = height; j > 0; j--) {
        std::getline(pgm_file, reader);
        parser.clear();
        parser.str(reader);
        for (size_t i = 0; i < width; i++) {
            parser >> data(i, j - 1);
            data(i, j - 1) = data(i, j - 1) / color_range;
        }
    }
    return data;
}

void write_pgm(std::ostream& pgm_file, Eigen::Ref<const Eigen::MatrixXd> image, size_t color_range = 255)
{
    // write pgm header
    pgm_file << "P2\n" << image.rows() << " " << image.cols() << "\n" << color_range << "\n";
    // write image data
    // we assume (0,0) to be at the bottom left
    const auto max = image.maxCoeff();
    for (Eigen::Index j = image.rows(); j > 0; j--) {
        for (Eigen::Index i = 0; i < image.cols(); i++) {
            pgm_file << size_t(image(i, j - 1) / max * color_range);
            if (i != image.cols() - 1)
                pgm_file << " ";
        }
        pgm_file << "\n";
    }
}

typedef Eigen::Vector2d Position;

// Add neighbouring (matrix) indices to queue
void enqueue_neighbours_if(const std::pair<ScalarType, ScalarType>& index, const std::pair<ScalarType, ScalarType>& max,
                           std::deque<std::pair<size_t, size_t>>& queue,
                           const std::function<bool(size_t, size_t)>& predicate)
{
    if (index.first > 0 && predicate(index.first - 1, index.second)) {
        queue.push_back({index.first - 1, index.second});
    }
    if (index.first + 1 < max.first && predicate(index.first + 1, index.second)) {
        queue.push_back({index.first + 1, index.second});
    }
    if (index.second > 0 && predicate(index.first, index.second - 1)) {
        queue.push_back({index.first, index.second - 1});
    }
    if (index.second + 1 < max.second && predicate(index.first, index.second + 1)) {
        queue.push_back({index.first, index.second + 1});
    }
}

// Add neighbouring (matrix) indices to queue
void enqueue_neighbours(const std::pair<ScalarType, ScalarType>& index, const std::pair<ScalarType, ScalarType>& max,
                        Eigen::Ref<Eigen::MatrixXd> mask, std::deque<std::pair<size_t, size_t>>& queue)
{
    // mask encoding : -1 - not , 1 - identified
    if (index.first > 0 && mask(index.first - 1, index.second) == 0.0) {
        mask(index.first - 1, index.second) = -1.0;
        queue.push_back({index.first - 1, index.second});
    }
    if (index.first + 1 < max.first && mask(index.first + 1, index.second) == 0.0) {
        mask(index.first + 1, index.second) = -1.0;
        queue.push_back({index.first + 1, index.second});
    }
    if (index.second > 0 && mask(index.first, index.second - 1) == 0.0) {
        mask(index.first, index.second - 1) = -1.0;
        queue.push_back({index.first, index.second - 1});
    }
    if (index.second + 1 < max.second && mask(index.first, index.second + 1) == 0.0) {
        mask(index.first, index.second + 1) = -1.0;
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
Eigen::MatrixXd find_connected_image_region(Eigen::Ref<const Eigen::MatrixXd> image, const size_t start_x,
                                            const size_t start_y, const ScalarType color_tolerance = 0)
{
    // setup
    size_t x, y;
    ScalarType color = image(start_x, start_y);
    std::deque<std::pair<size_t, size_t>> queue({{start_x, start_y}});
    Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(image.rows(), image.cols());
    // iterate (BFS)
    for (; not queue.empty(); queue.pop_front()) {
        std::tie(x, y) = queue.front();
        if (std::abs(color - image(x, y)) <= color_tolerance) {
            mask(x, y) = 1.0;
            enqueue_neighbours({x, y}, {image.rows(), image.cols()}, mask, queue);
        }
        else {
            mask(x, y) = 0.0;
        }
    }
    return mask;
}

// Eigen::MatrixXd find_bounded_image_region(Eigen::Ref<Eigen::MatrixXd> image, const size_t start_x, const size_t start_y,
//                                           ScalarType boundary_color, ScalarType color_tolerance = 0);

void apply_stencil(Eigen::Ref<Eigen::MatrixXd> image, Eigen::Ref<const Eigen::MatrixXd> stencil, ScalarType color,
                   ScalarType color_tolerance = 0)
{
    const Eigen::Index rows = stencil.rows();
    const Eigen::Index cols = stencil.cols();
    assert(rows > 0 && cols > 0 && "The stencil must have static size.");
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

std::vector<Position> laender = {
    {0.16458333333333333, 0.4479166666666667}, // Schleswig-Holstein
    {0.21666666666666667, 0.6333333333333333}, // MVP
    {0.23958333333333334, 0.46458333333333335}, // Hamburg
    {0.28958333333333336, 0.38958333333333334}, // Bremen
    {0.3020833333333333, 0.4479166666666667}, // Niedersachsen
    {0.30416666666666664, 0.6916666666666667}, // Brandenburg
    {0.34375, 0.69375}, // Berlin
    {0.4270833333333333, 0.58125}, // Sachsen Anhalt
    {0.46875, 0.2916666666666667}, // NRW
    {0.5083333333333333, 0.6979166666666666}, // Sachsen
    {0.525, 0.5354166666666667}, // Thüringen
    {0.5895833333333333, 0.38958333333333334}, // Hessen
    {0.6354166666666666, 0.28541666666666665}, // RLP
    {0.69375, 0.25416666666666665}, // Saarland
    {0.74375, 0.6104166666666667}, // Bayern
    {0.8, 0.4}, // BaWü
};

int main()
{
    DEBUG("open file")
    std::ifstream file("potential_dpi=300.pgm");
    DEBUG("read file")
    auto image = read_pgm(file);
    file.close();
    image = (1 - image.array()).matrix(); // invert colors

    DEBUG("map federal states")
    size_t i               = 0;
    Eigen::MatrixXd canvas = Eigen::MatrixXd::Zero(image.rows(), image.cols());
    for (auto land : laender) {
        ++i;
        const auto x = land[1] * canvas.rows();
        const auto y = (1 - land[0]) * canvas.cols();
        canvas += i * find_connected_image_region(image, x, y, 0);
    }

    DEBUG("set stencil")
    Eigen::Matrix<double, 1, 5> stencil;
    stencil(0, 0) = 0.5;
    stencil(0, 1) = 1.0;
    stencil(0, 2) = 1.0;
    stencil(0, 3) = 1.0;
    stencil(0, 4) = 0.5;

    Eigen::Matrix<double, 1, 5> stencil_ext;
    stencil_ext(0, 0) = 0.5;
    stencil_ext(0, 1) = 0.75;
    stencil_ext(0, 2) = 1.0;
    stencil_ext(0, 3) = 0.75;
    stencil_ext(0, 4) = 0.5;

    // DEBUG(stencil.transpose() * stencil);

    DEBUG("apply stencil")
    apply_stencil(image, stencil.transpose() * stencil, 1.0);

    DEBUG("expand border")
    auto exterior = find_connected_image_region(image, 0, 0, 0.75);
    apply_stencil(exterior, stencil_ext.transpose() * stencil_ext, 1.0);
    image = image.array().max(4 * exterior.array()).matrix();
    // image = 4 * exterior;

    DEBUG("write files")
    std::ofstream ofile("potentially_germany.pgm");
    if (!ofile.is_open()) {
        DEBUG("Could not open file potentially_germany.pgm");
        return 1;
    }
    write_pgm(ofile, image);
    ofile.close();
    ofile.open("metagermany.pgm");
    if (!ofile.is_open()) {
        DEBUG("Could not open file metagermany.pgm");
        return 1;
    }
    write_pgm(ofile, canvas, 16);
    ofile.close();

    // DEBUG("print")
    // for (Eigen::Index j = 0; j < image.cols(); j++) {
    //     for (Eigen::Index i = 0; i < image.rows(); i++) {
    //         std::cout << image(i, image.cols() - 1 - j) << " ";
    //     }
    //     std::cout << "\n";
    // }
    DEBUG("success")
}
