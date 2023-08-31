#ifndef MAP_READER_H_
#define MAP_READER_H_

#include "memilio/math/eigen.h"

#include <deque>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

#define DEBUG(cout_args) std::cerr << cout_args << std::endl << std::flush;

namespace mio
{
namespace mpm
{

std::pair<Eigen::MatrixXi, size_t> read_pgm_raw(std::istream& pgm_file);

Eigen::MatrixXd read_pgm(std::istream& pgm_file);

void write_pgm(std::ostream& pgm_file, Eigen::Ref<const Eigen::MatrixXd> image, size_t color_range = 255);

void write_pgm(std::ostream& pgm_file, Eigen::Ref<const Eigen::MatrixXi> image);

template <class Integer>
size_t num_bits_set(Integer i)
{
    size_t c;
    for (c = 0; i; c++) {
        i &= i - 1; // removes rightmost 1 from i
    }
    return c;
}

} // namespace mpm
} // namespace mio

#endif // MAP_READER_H_