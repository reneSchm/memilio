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

typedef Eigen::Vector2d Position;

ScalarType interpolate_bilinear(Eigen::Ref<const Eigen::MatrixXd> data, const Position& p)
{
    size_t x = p[0], y = p[1]; // take floor of each coordinate
    const ScalarType tx = p[0] - x;
    const ScalarType ty = p[1] - y;
    // data point distance is always one, since we use integer indices
    const ScalarType bot = tx * data(x, y) + (1 - tx) * data(x + 1, y);
    const ScalarType top = tx * data(x, y + 1) + (1 - tx) * data(x + 1, y + 1);
    return ty * bot + (1 - ty) * top;
}

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
    size_t x, y, qm = 0;
    ScalarType color = image(start_x, start_y);
    std::deque<std::pair<size_t, size_t>> queue({{start_x, start_y}});
    Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(image.rows(), image.cols());
    // iterate (BFS)
    for (; not queue.empty(); queue.pop_front()) {
        std::tie(x, y) = queue.front();
        qm             = std::max(qm, queue.size());
        if (std::abs(color - image(x, y)) <= color_tolerance) {
            mask(x, y) = 1.0;
            enqueue_neighbours({x, y}, {image.rows(), image.cols()}, mask, queue);
        }
        else {
            mask(x, y) = 0.0;
        }
    }
    std::cerr << qm << "\n";
    return mask;
}

// Eigen::MatrixXd find_bounded_image_region(Eigen::Ref<Eigen::MatrixXd> image, const size_t start_x, const size_t start_y,
//                                           ScalarType boundary_color, ScalarType color_tolerance = 0);

std::vector<Position> laender = {
    {0.16458333333333333, 0.4479166666666667}, // BaWü
    {0.21666666666666667, 0.6333333333333333}, // Bayern
    {0.23958333333333334, 0.46458333333333335},
    {0.28958333333333336, 0.38958333333333334},
    {0.3020833333333333, 0.4479166666666667},
    {0.30416666666666664, 0.6916666666666667},
    {0.34375, 0.69375},
    {0.4270833333333333, 0.58125},
    {0.46875, 0.2916666666666667},
    {0.5083333333333333, 0.6979166666666666},
    {0.525, 0.5354166666666667},
    {0.5895833333333333, 0.38958333333333334},
    {0.6354166666666666, 0.28541666666666665},
    {0.69375, 0.25416666666666665},
    {0.74375, 0.6104166666666667},
    {0.8, 0.4},
};

int main()
{
    std::ifstream file("potential_dpi=100.pgm");
    auto image = read_pgm(file);

    int land = 0;
    auto x   = laender[land][1] * image.rows();
    auto y   = laender[land][0] * image.cols();
    image += 8 * find_connected_image_region(image, x, y, 0);

    for (Eigen::Index j = 0; j < image.cols(); j++) {
        for (Eigen::Index i = 0; i < image.rows(); i++) {
            std::cout << image(i, image.cols() - 1 - j) << " ";
        }
        std::cout << "\n";
    }
}
