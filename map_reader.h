#ifndef MAP_READER_H_
#define MAP_READER_H_

#include <deque>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#include "build/_deps/eigen-src/Eigen/Core"
#define DEBUG(cout_args) std::cerr << cout_args << std::endl << std::flush;
typedef double ScalarType;

std::pair<Eigen::MatrixXi, size_t> read_pgm_raw(std::istream& pgm_file)
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
    Eigen::MatrixXi data(width, height);
    for (size_t j = height; j > 0; j--) {
        std::getline(pgm_file, reader);
        parser.clear();
        parser.str(reader);
        for (size_t i = 0; i < width; i++) {
            parser >> data(i, j - 1);
        }
    }
    return std::make_pair(data, color_range);
}

Eigen::MatrixXd read_pgm(std::istream& pgm_file)
{
    auto raw = read_pgm_raw(pgm_file);
    return raw.first.cast<ScalarType>() / raw.second; // convert matrix and normalize color range
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

void write_pgm(std::ostream& pgm_file, Eigen::Ref<const Eigen::MatrixXi> image)
{
    // write pgm header
    const auto color_range = image.maxCoeff();
    pgm_file << "P2\n" << image.rows() << " " << image.cols() << "\n" << color_range << "\n";
    // write image data
    // we assume (0,0) to be at the bottom left
    for (Eigen::Index j = image.rows(); j > 0; j--) {
        for (Eigen::Index i = 0; i < image.cols(); i++) {
            pgm_file << size_t(image(i, j - 1));
            if (i != image.cols() - 1)
                pgm_file << " ";
        }
        pgm_file << "\n";
    }
}

template <class Integer>
size_t num_bits_set(Integer i)
{
    size_t c;
    for (c = 0; i; c++) {
        i &= i - 1; // removes rightmost 1 from i
    }
    return c;
}

#endif // MAP_READER_H_