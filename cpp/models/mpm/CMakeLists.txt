add_library(mpm
    abm.h
    model.cpp
    model.h
    pdmm.h
    region.h
    smm.h
    utility.cpp
    utility.h
    potentials/potential_germany.h
)
target_link_libraries(mpm PUBLIC memilio)
target_include_directories(mpm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_compile_options(mpm PRIVATE ${MEMILIO_CXX_FLAGS_ENABLE_WARNING_ERRORS})
