# ============================================================================
# CS3006 — Parallel & Distributed Computing
# Project #16: Parallel Genetic Algorithm for Large Search Spaces
# Hybrid MPI + OpenMP implementation for the Traveling Salesman Problem
# ============================================================================
cmake_minimum_required(VERSION 3.20)

project(pdc_parallel_ga
    VERSION 0.1.0
    DESCRIPTION "Hybrid MPI+OpenMP Parallel Genetic Algorithm for TSP"
    LANGUAGES CXX
)

# ---- C++ Standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Build Type Default ----
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---- Compiler Flags ----
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG   "-O0 -g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -march=native")

# ---- ccache (auto-use if installed) ----
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    message(STATUS "ccache found: ${CCACHE_PROGRAM} — compilation will be cached")
endif()

# ---- Find Packages ----
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

# ---- Output Directory ----
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/bin)

# ---- Main Executable ----
add_executable(pdc_ga
    src/main.cpp
    src/ga.cpp
    src/fitness.cpp
    src/mpi_manager.cpp
    src/utils.cpp
)
target_include_directories(pdc_ga PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(pdc_ga PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)

# ---- Enable Testing ----
enable_testing()
add_subdirectory(tests)