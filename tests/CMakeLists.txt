# ============================================================================
# Unit tests and standalone harnesses
# ============================================================================

# ---- Basic unit tests ----
# The test file calls functions defined in src/ga.cpp and src/fitness.cpp,
# so those translation units must be compiled into the test executable.
add_executable(pdc_tests
    test_ga.cpp
    ${CMAKE_SOURCE_DIR}/src/ga.cpp
    ${CMAKE_SOURCE_DIR}/src/fitness.cpp
)
target_include_directories(pdc_tests PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(pdc_tests PRIVATE MPI::MPI_CXX OpenMP::OpenMP_CXX)
add_test(NAME basic COMMAND pdc_tests)

# ---- Fitness cross-check harness ----
# Standalone binary that loads a coords file + a tour file, builds the
# distance matrix, computes the tour length, and prints it. Used to verify
# tour_length() against a NumPy reference implementation.
add_executable(fitness_check
    fitness_check.cpp
    ${CMAKE_SOURCE_DIR}/src/fitness.cpp
)
target_include_directories(fitness_check PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(fitness_check PRIVATE OpenMP::OpenMP_CXX)