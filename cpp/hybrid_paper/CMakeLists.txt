add_subdirectory(dlib-19.24 dlib_build)

add_library(hybrid_paper initialization.h initialization.cpp infection_state.h)
target_link_libraries(hybrid_paper memilio mpm)

# find_package(OpenMP REQUIRED)
add_executable(transition_rate_fitting transition_rate_fitting.cpp)
target_link_libraries(transition_rate_fitting dlib::dlib hybrid_paper)

add_executable(estimate_transition_rates estimate_transition_rates.cpp)
target_link_libraries(estimate_transition_rates hybrid_paper)

add_executable(create_agents create_agents.cpp)
target_link_libraries(create_agents hybrid_paper)

add_executable(adoption_rate_fitting adoption_rate_fitting.cpp)
target_link_libraries(adoption_rate_fitting dlib::dlib hybrid_paper)

add_executable(map_reader map_reader.cpp)
target_link_libraries(map_reader hybrid_paper)

add_executable(mpm_abm_sim mpm_abm_sim.cpp)
target_link_libraries(mpm_abm_sim hybrid_paper)

add_executable(test test.cpp)
target_link_libraries(test hybrid_paper)

add_executable(k-model k-model.cpp)
target_link_libraries(k-model hybrid_paper)

add_executable(transition_rate_fitting_random_search transition_rate_fitting_random_search.cpp)
target_link_libraries(transition_rate_fitting_random_search hybrid_paper)

add_executable(contact_frequency_counting contact_frequency_counting.cpp)
target_link_libraries(contact_frequency_counting hybrid_paper)
