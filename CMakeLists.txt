cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(hipRAND CXX C)

# CMake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Options
option(BUILD_SHARED_LIBS "Build libhiprand.so" ON)
# Include cmake scripts
include(cmake/Summary.cmake)
include(cmake/Dependencies.cmake)

# Build flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -fPIC -Wno-narrowing")

# Print configuration summary
print_configuration_summary()

# hipRAND library
# This adds library target hipRAND
add_subdirectory(library)
