#include "memilio/io/mobility_io.h"
#include "memilio/compartments/parameter_studies.h"
#include "memilio/io/epi_data.h"
#include "memilio/io/result_io.h"
#include "memilio/io/mobility_io.h"
#include "memilio/utils/date.h"
#include "memilio/utils/random_number_generator.h"
#include "ode_secirvvs/parameters_io.h"
#include "ode_secirvvs/parameter_space.h"
#include "memilio/utils/stl_util.h"
#include "boost/filesystem.hpp"
#include <cstdio>
#include <iomanip>

/**
 * Indices of the transition matrix correspond to the following federal states
 *  0: 'Schleswig-Holstein',
    1: 'Hamburg',
    2: 'Niedersachsen',
    3: 'Bremen',
    4: 'Nordrhein-Westfalen',
    5: 'Hessen',
    6: 'Rheinland-Pfalz',
    7: 'Baden-Württemberg',
    8: 'Bayern',
    9: 'Saarland',
    10: 'Berlin',
    11: 'Brandenburg',
    12: 'Mecklenburg-Vorpommern',
    13: 'Sachsen',
    14: 'Sachsen-Anhalt',
    15: 'Thüringen'
*/
mio::IOResult<Eigen::MatrixXd> get_transition_matrices(std::string data_dir)
{
    BOOST_OUTCOME_TRY(matrix_commuter, mio::read_mobility_plain(data_dir + "/commuter_migration_scaled_states.txt"));
    BOOST_OUTCOME_TRY(matrix_twitter, mio::read_mobility_plain(data_dir + "/twitter_scaled_1252_states.txt"));
    Eigen::MatrixXd travel_to_matrix = matrix_commuter + matrix_twitter;
    Eigen::MatrixXd transitions_per_day(travel_to_matrix.rows(), travel_to_matrix.cols());
    for (int from = 0; from < travel_to_matrix.rows(); ++from) {
        for (int to = 0; to < travel_to_matrix.cols(); ++to) {
            transitions_per_day(from, to) = travel_to_matrix(from, to) + travel_to_matrix(to, from);
        }
    }
    return mio::success(transitions_per_day);
}
/**
 * Parameter fitting
 * 1. get transitions per day from mobility matrices -> for one federal state i the number of transitions per day it sends to state j 
 * is the number it send to state j (m(i,j)) at the beginning of the day plus the number that state j had send to state i (m(j,i)) and that returns at the end of the day.
 * So  m(i, j) + m(j, i)
 * 
 * 2. get the transitions from simulation
 *  -> in the abm simulation the transitions will noch necessarily the same every day, so we need to multiply the transitions from the matrices
 * with the number of simulation days to be able to compare it
 * -> we should store the number of transitions matrix wise like the commuter matrices are given
 * 
 * 3. compare simualtion transitions with matrix transitions
 * -> what error metric? mean squared error? mean absolute error?
 * -> do we compare the number of transitions over the whole simulation or do we calculate the error for every day and average over all days
 * 
 * 4. parameter adjustment
 * -> find reasonable default values (intervals) for the transition rates
 * -> uniformly draw a values for the transition rates and make x (maybe 100?) abm runs with that and average over the number of transitions for every run
 * -> calculate error with this averaged number
 * -> repeat for x (100?) drawings and set the parameters to the values with the lowest error
 */

int main()
{
    std::string data_dir = "C:/Users/bick_ju/Documents/repos/hybrid/memilio/data/mobility";
    auto result          = get_transition_matrices(data_dir);
    return 0;
}