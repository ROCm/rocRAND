cmake_minimum_required(VERSION 3.0.2 FATAL_ERROR)

# Install prefix
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

# CMake modules
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
    ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

# Identify HIP platform (nvcc or hcc), and select C++ compiler:
# ROCm - hcc
# CUDA - g++ (nvcc is used via functions from CUDA package)
include(cmake/SetToolchain.cmake)

#
# rocRAND project
#
project(rocRAND CXX)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOLEAN "Add paths to linker search and installed rpath")

# Build CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-variadic-macros -Wno-sign-compare")

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_FORTRAN_WRAPPER "Build Fortran wrapper" OFF)
option(BUILD_TEST "Build tests (requires googletest)" ON)
option(BUILD_BENCHMARK "Build benchmarks" OFF)

# Include cmake scripts
include(cmake/Dependencies.cmake)
include(cmake/Summary.cmake)

# Print configuration summary
print_configuration_summary()

# Set version variables
include(cmake/Common.cmake)
create_version_vars(hiprand 1 6 1)
create_version_vars(rocrand 1 6 1)

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900 CACHE STRING "List of specific machine types for library to target")

# Tools
add_subdirectory(tools)

# rocRAND and hipRAND libraries
# This adds library targets: rocrand, hiprand,
# also includes Fortran wrapper
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
