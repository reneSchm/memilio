#include "hybrid_paper/initialization.h"
#include "hybrid_paper/infection_state.h"
#include "mpm/pdmm.h"
#include "mpm/region.h"
#include "memilio/data/analyze_result.h"
#include "memilio/io/epi_data.h"
#include "memilio/io/cli.h"

#define TIME_TYPE std::chrono::high_resolution_clock::time_point
#define TIME_NOW std::chrono::high_resolution_clock::now()
#define PRINTABLE_TIME(_time) (std::chrono::duration_cast<std::chrono::duration<double>>(_time)).count()
#define PRECISION 17

#define restart_timer(timer, description)                                                                              \
    {                                                                                                                  \
        TIME_TYPE new_time = TIME_NOW;                                                                                 \
        std::cout << "\r" << description << " :: " << PRINTABLE_TIME(new_time - timer) << std::endl << std::flush;     \
        timer = TIME_NOW;                                                                                              \
    }

struct TransmissionRate {
    using Type = double;
    const static std::string name()
    {
        return "TransmissionRate";
    }
    const static std::string alias()
    {
        return "tr";
    }
};

int main(int argc, char** argv)
{
    mio::set_log_level(mio::LogLevel::warn);
    using Status = mio::mpm::paper::InfectionState;
    using Region = mio::mpm::Region;
    using Model  = mio::mpm::PDMModel<8, Status>;
    //number inhabitants per region
    std::vector<double> populations = {218579, 155449, 136747, 1487708, 349837, 181144, 139622, 144562};
    //county ids
    std::vector<int> regions = {9179, 9174, 9188, 9162, 9184, 9178, 9177, 9175};

    //estimated transition rates
    std::map<std::tuple<Region, Region>, double> factors{
        {{Region(0), Region(1)}, 0.000958613}, {{Region(0), Region(2)}, 0.00172736},
        {{Region(0), Region(3)}, 0.0136988},   {{Region(0), Region(4)}, 0.00261568},
        {{Region(0), Region(5)}, 0.000317227}, {{Region(0), Region(6)}, 0.000100373},
        {{Region(0), Region(7)}, 0.00012256},  {{Region(1), Region(0)}, 0.000825387},
        {{Region(1), Region(2)}, 0.00023648},  {{Region(1), Region(3)}, 0.0112213},
        {{Region(1), Region(4)}, 0.00202101},  {{Region(1), Region(5)}, 0.00062912},
        {{Region(1), Region(6)}, 0.000201067}, {{Region(1), Region(7)}, 0.000146773},
        {{Region(2), Region(0)}, 0.000712533}, {{Region(2), Region(1)}, 0.000102613},
        {{Region(2), Region(3)}, 0.00675979},  {{Region(2), Region(4)}, 0.00160171},
        {{Region(2), Region(5)}, 0.000175467}, {{Region(2), Region(6)}, 0.00010336},
        {{Region(2), Region(7)}, 6.21867e-05}, {{Region(3), Region(0)}, 0.00329632},
        {{Region(3), Region(1)}, 0.00322347},  {{Region(3), Region(2)}, 0.00412565},
        {{Region(3), Region(4)}, 0.0332566},   {{Region(3), Region(5)}, 0.00462197},
        {{Region(3), Region(6)}, 0.00659424},  {{Region(3), Region(7)}, 0.00255147},
        {{Region(4), Region(0)}, 0.000388373}, {{Region(4), Region(1)}, 0.000406827},
        {{Region(4), Region(2)}, 0.000721387}, {{Region(4), Region(3)}, 0.027394},
        {{Region(4), Region(5)}, 0.00127328},  {{Region(4), Region(6)}, 0.00068224},
        {{Region(4), Region(7)}, 0.00104491},  {{Region(5), Region(0)}, 0.00013728},
        {{Region(5), Region(1)}, 0.000475627}, {{Region(5), Region(2)}, 0.00010688},
        {{Region(5), Region(3)}, 0.00754293},  {{Region(5), Region(4)}, 0.0034704},
        {{Region(5), Region(6)}, 0.00210027},  {{Region(5), Region(7)}, 0.000226667},
        {{Region(6), Region(0)}, 7.264e-05},   {{Region(6), Region(1)}, 0.0001424},
        {{Region(6), Region(2)}, 9.55733e-05}, {{Region(6), Region(3)}, 0.00921109},
        {{Region(6), Region(4)}, 0.0025216},   {{Region(6), Region(5)}, 0.00266944},
        {{Region(6), Region(7)}, 0.00156053},  {{Region(7), Region(0)}, 7.81867e-05},
        {{Region(7), Region(1)}, 0.0001024},   {{Region(7), Region(2)}, 8.256e-05},
        {{Region(7), Region(3)}, 0.00833152},  {{Region(7), Region(4)}, 0.00393717},
        {{Region(7), Region(5)}, 0.000354987}, {{Region(7), Region(6)}, 0.00055456}};

    std::vector<Status> transitioning_states{Status::S, Status::E, Status::C, Status::I, Status::R};
    //No adapted transition behaviour when infected
    std::vector<mio::mpm::TransitionRate<Status>> transition_rates;
    for (auto& rate : factors) {
        for (auto s : transitioning_states) {
            transition_rates.push_back({s, std::get<0>(rate.first), std::get<1>(rate.first), rate.second});
        }
    }
    std::vector<mio::ConfirmedCasesDataEntry> confirmed_cases =
        mio::read_confirmed_cases_data(mio::base_dir() + "/data/Germany/cases_all_county_age_ma7.json").value();

    std::sort(confirmed_cases.begin(), confirmed_cases.end(), [](auto&& a, auto&& b) {
        return std::make_tuple(get_region_id(a), a.date) < std::make_tuple(get_region_id(b), b.date);
    });

    auto cli_result = mio::command_line_interface<TransmissionRate>(argv[0], argc, argv);
    if (!cli_result) {
        std::cerr << cli_result.error().formatted_message() << "\n";
        return cli_result.error().code().value();
    }

    const double tmax        = 30;
    double dt                = 0.1;
    mio::Date start_date     = mio::Date(2021, 3, 1);
    double t_Exposed         = 4;
    double t_Carrier         = 2.67;
    double t_Infected        = 5.03;
    double mu_C_R            = 0.29;
    double transmission_rate = cli_result.value().get<TransmissionRate>(); //0.25;
    double mu_I_D            = 0.00476;

    Model model;

    //vector with entry for every region. Entries are vector with population for every infection state according to initialization
    std::vector<std::vector<double>> pop_dists =
        set_confirmed_case_data(confirmed_cases, regions, populations, start_date, t_Exposed, t_Carrier, t_Infected,
                                mu_C_R)
            .value();

    //set populations for model
    for (size_t k = 0; k < regions.size(); ++k) {
        for (int i = 0; i < static_cast<size_t>(Status::Count); ++i) {
            model.populations[{static_cast<mio::mpm::Region>(k), static_cast<Status>(i)}] = pop_dists[k][i];
        }
    }
    //set transion rates for model (according to parameter estimation)
    model.parameters.get<mio::mpm::TransitionRates<Status>>() = transition_rates;
    //set adoption rates according to given parameters
    std::vector<mio::mpm::AdoptionRate<Status>> adoption_rates;
    for (int i = 0; i < regions.size(); ++i) {
        adoption_rates.push_back(
            {Status::S, Status::E, mio::mpm::Region(i), transmission_rate, {Status::C, Status::I}, {1, 1}});
        adoption_rates.push_back({Status::E, Status::C, mio::mpm::Region(i), 1.0 / t_Exposed});
        adoption_rates.push_back({Status::C, Status::R, mio::mpm::Region(i), mu_C_R / t_Carrier});
        adoption_rates.push_back({Status::C, Status::I, mio::mpm::Region(i), (1 - mu_C_R) / t_Carrier});
        adoption_rates.push_back({Status::I, Status::R, mio::mpm::Region(i), (1 - mu_I_D) / t_Infected});
        adoption_rates.push_back({Status::I, Status::D, mio::mpm::Region(i), mu_I_D / t_Infected});
    }
    model.parameters.get<mio::mpm::AdoptionRates<Status>>() = adoption_rates;
    auto result                                             = mio::simulate(0, tmax, dt, model);
    auto interpolated_result                                = mio::interpolate_simulation_result(result);

    std::string dir = mio::base_dir() + "cpp/outputs/";
    FILE* out_file  = fopen((dir + "PDMM_output.txt").c_str(), "w");
    mio::mpm::print_to_file(out_file, interpolated_result, {});
    fclose(out_file);

    return 0;
}
