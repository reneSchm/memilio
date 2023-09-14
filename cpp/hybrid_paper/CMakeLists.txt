add_subdirectory(dlib-19.24 dlib_build)

add_executable(fitting_trial fitting_trial.cpp)
target_link_libraries(fitting_trial dlib::dlib memilio mpm)

add_executable(dlib_example dlib_example.cpp)
target_link_libraries(dlib_example dlib::dlib)