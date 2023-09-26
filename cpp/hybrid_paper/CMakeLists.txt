add_subdirectory(dlib-19.24 dlib_build)

add_executable(fitting_trial fitting_trial.cpp)
target_link_libraries(fitting_trial dlib::dlib memilio mpm)

add_executable(dlib_example dlib_example.cpp)
target_link_libraries(dlib_example dlib::dlib)

add_executable(estimate_transition_rates estimate_transition_rates.cpp)
target_link_libraries(estimate_transition_rates memilio mpm)

add_executable(initialization initialization)
target_link_libraries(initialization memilio mpm)