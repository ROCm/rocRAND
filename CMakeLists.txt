cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(hipRAND CXX C)

# CMake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Options
option(BUILD_SHARED_LIBS "Build libhiprand.so" ON)
option(BUILD_TEST "Build tests (requires googletest)" ON)
option(BUILD_BENCHMARK "Build benchmarks" OFF)

# Include cmake scripts
include(cmake/Summary.cmake)
include(cmake/Dependencies.cmake)

# Build flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -fPIC -Wno-narrowing")
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic -Werror -Wno-variadic-macros")

# Print configuration summary
print_configuration_summary()

add_subdirectory(tools)

# rocRAND library
# This adds library target rocRAND
add_subdirectory(library)

# Tests
if (BUILD_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

# Benchmarks
if (BUILD_BENCHMARK)
    add_subdirectory(benchmark)
endif()
