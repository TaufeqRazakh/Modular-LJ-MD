cmake_minimum_required(VERSION 3.12)
project(Modular-MD)

set(CMAKE_CXX_STANDARD 11)

add_library(pmd STATIC
        pmd.cpp
        pmd.hpp
        )

find_package(MPI REQUIRED)

include_directories(${MPI_INCLUDE_PATH})

add_executable(Modular-MD pmd_notebook.cpp)
target_link_libraries(Modular-MD pmd)

target_link_libraries(Modular-MD ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
    set_target_properties(Modular-MD PROPERTIES
            COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

#if(MPI_LINK_FLAGS)
#    set_target_properties(Molecular-MD PROPERTIES
#            LINK_FLAGS "${MPI_LINK_FLAGS}")
#endif()
