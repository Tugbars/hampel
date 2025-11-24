cmake_minimum_required(VERSION 3.15)
project(HampelFilterProduction C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Compiler flags for optimization
if(MSVC)
    add_compile_options(/W4 /arch:AVX2 /O2 /fp:fast)
else()
    add_compile_options(-Wall -Wextra -mavx2 -O3 -ffast-math -march=native)
endif()

# Optional: Disable diagnostics for maximum performance
option(HAMPEL_NO_DIAGNOSTICS "Disable diagnostic counters for max performance" OFF)
if(HAMPEL_NO_DIAGNOSTICS)
    add_definitions(-DHAMPEL_NO_DIAGNOSTICS)
endif()

# Header-only library
add_library(hampel_robust INTERFACE)
target_include_directories(hampel_robust INTERFACE include)

# Test executable
add_executable(test_hampel tests/test_hampel.c)
target_link_libraries(test_hampel hampel_robust m)

# Example executable
add_executable(example_usage examples/example_usage.c)
target_link_libraries(example_usage hampel_robust m)

# Enable testing
enable_testing()
add_test(NAME HampelTests COMMAND test_hampel)

# Installation
install(DIRECTORY include/ DESTINATION include)