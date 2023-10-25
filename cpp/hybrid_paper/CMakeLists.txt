add_subdirectory(dlib-19.24 dlib_build)

find_package(OpenMP REQUIRED)
add_executable(transition_rate_fitting transition_rate_fitting.cpp)
target_link_libraries(transition_rate_fitting dlib::dlib OpenMP::OpenMP_CXX memilio mpm)

add_executable(estimate_transition_rates estimate_transition_rates.cpp)
target_link_libraries(estimate_transition_rates memilio mpm)

add_executable(initialization initialization)
target_link_libraries(initialization memilio mpm)

add_executable(adoption_rate_fitting adoption_rate_fitting.cpp)
target_link_libraries(adoption_rate_fitting dlib::dlib memilio mpm)

add_executable(map_reader map_reader.cpp)
target_link_libraries(map_reader memilio mpm)
