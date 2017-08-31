cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

project(hipRAND CXX C)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "${Release}" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# CMake modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# Options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TEST "Build tests (requires googletest)" ON)
option(BUILD_BENCHMARK "Build benchmarks" OFF)

# Version vars
include(cmake/Common.cmake)
create_version_vars(hiprand 1 5 0)
create_version_vars(rocrand 1 5 0)

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900 CACHE STRING "List of specific machine types for library to target")

# Include cmake scripts
include(cmake/Dependencies.cmake)
include(cmake/Summary.cmake)

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
