/* 
* Copyright (C) 2020-2023 German Aerospace Center (DLR-SC)
*
* Authors: René Schmieding
*
* Contact: Martin J. Kuehn <Martin.Kuehn@DLR.de>
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "mpm/utility.h"
#include <stdio.h>

void mio::mpm::print_to_terminal(const mio::TimeSeries<ScalarType>& results,
                                 const std::vector<std::string>& state_names)
{
    mio::mpm::print_to_file(stdout, results, state_names);
}

void mio::mpm::print_to_file(FILE* outfile, const mio::TimeSeries<ScalarType>& results,
                             const std::vector<std::string>& state_names)
{
    // print column labels
    fprintf(outfile, "%-16s  ", "Time");
    for (size_t k = 0; k < static_cast<size_t>(results.get_num_elements()); k++) {
        if (k < state_names.size()) {
            fprintf(outfile, " %-16s", state_names[k].data()); // print underlying char*
        }
        else {
            fprintf(outfile, " %-16s", ("#" + std::to_string(k + 1)).data());
        }
    }
    // print values as table
    auto num_points = static_cast<size_t>(results.get_num_time_points());
    for (size_t i = 0; i < num_points; i++) {
        fprintf(outfile, "\n%16.6f", results.get_time(i));
        auto res_i = results.get_value(i);
        for (size_t j = 0; j < static_cast<size_t>(res_i.size()); j++) {
            fprintf(outfile, " %16.6f", res_i[j]);
        }
    }
    fprintf(outfile, "\n");
}