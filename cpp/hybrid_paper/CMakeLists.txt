add_subdirectory(dlib-19.24 dlib_build)

add_library(hybrid_paper
    library/ensemble_run.h
    library/ensemble_run.cpp
    library/initialization.h
    library/initialization.cpp
    library/infection_state.h
    library/infection_state.cpp
    library/map_reader.h
    library/map_reader.cpp
    library/metaregion_sampler.h
    library/model_setup.h
    library/potentials/potential_germany.h
    library/potentials/commuting_potential.h
    library/transition_rate_estimation.h
    library/transition_rate_estimation.cpp
    library/weighted_gradient.h
    library/weighted_potential.h
    quad_well/quad_well_setup.h
)
target_link_libraries(hybrid_paper memilio mpm)

# target_include_directories(hybrid_paper PUBLIC
# $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
# $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
# )
target_compile_options(hybrid_paper PRIVATE ${MEMILIO_CXX_FLAGS_ENABLE_WARNING_ERRORS})

add_executable(transition_rate_fitting munich/transition_rate_fitting.cpp)
target_link_libraries(transition_rate_fitting dlib::dlib hybrid_paper)

add_executable(estimate_transition_rates_munich munich/estimate_transition_rates.cpp)
target_link_libraries(estimate_transition_rates_munich hybrid_paper)

add_executable(estimate_transition_rates_quadwell quad_well/estimate_transition_rates.cpp)
target_link_libraries(estimate_transition_rates_quadwell hybrid_paper)

add_executable(create_agents munich/create_agents.cpp)
target_link_libraries(create_agents hybrid_paper)

add_executable(adoption_rate_fitting munich/adoption_rate_fitting.cpp)
target_link_libraries(adoption_rate_fitting dlib::dlib hybrid_paper)

add_executable(map_reader munich/map_reader.cpp)
target_link_libraries(map_reader hybrid_paper)

find_package(OpenMP REQUIRED)
add_executable(mpm_abm_sim munich/mpm_abm_sim.cpp)
target_link_libraries(mpm_abm_sim hybrid_paper OpenMP::OpenMP_CXX)

add_executable(test test.cpp)
target_link_libraries(test hybrid_paper)

add_executable(k-model k-model.cpp)
target_link_libraries(k-model hybrid_paper)

add_executable(transition_rate_fitting_random_search munich/transition_rate_fitting_random_search.cpp)
target_link_libraries(transition_rate_fitting_random_search hybrid_paper)

add_executable(contact_frequency_counting munich/contact_frequency_counting.cpp)
target_link_libraries(contact_frequency_counting hybrid_paper)

add_executable(hybridization munich/hybridization.cpp)
target_link_libraries(hybridization hybrid_paper OpenMP::OpenMP_CXX)

add_executable(PDMM munich/PDMM.cpp)
target_link_libraries(PDMM hybrid_paper)

add_executable(ABM_PDMM_munich munich/ABM_PDMM.cpp)
target_link_libraries(ABM_PDMM_munich hybrid_paper)

add_executable(ABM_PDMM_quad_well quad_well/ABM_PDMM.cpp)
target_link_libraries(ABM_PDMM_quad_well hybrid_paper)

add_executable(travel_time_heuristic munich/travel_time_heuristic.cpp)
target_link_libraries(travel_time_heuristic hybrid_paper)

add_executable(spatial_hybridization_quad_well quad_well/spatial_hybridization.cpp)
target_link_libraries(spatial_hybridization_quad_well hybrid_paper OpenMP::OpenMP_CXX)

add_executable(qw_temporal_hybrid quad_well/temporal_hybrid.cpp)
target_link_libraries(qw_temporal_hybrid hybrid_paper)
