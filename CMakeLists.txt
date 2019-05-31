# MIT License
#
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

# Install prefix
set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

# CMake modules
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
    ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
)

#
# rocRAND project
#
project(rocrand CXX)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOLEAN "Add paths to linker search and installed rpath")

# Verify that hcc compiler is used on ROCM platform
include(cmake/VerifyCompiler.cmake)

# Build CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_FORTRAN_WRAPPER "Build Fortran wrapper" OFF)
option(BUILD_TEST "Build tests (requires googletest)" ON)
option(BUILD_CRUSH_TEST "Build Crush tests (requires TestU01)" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_TOOLS "Build tools" OFF)

# Include cmake scripts
include(cmake/Dependencies.cmake)
include(cmake/Summary.cmake)

# Print configuration summary
print_configuration_summary()

# Setup VERSION
set(VERSION_STRING "2.5.0.")

# Check if BUILD_NUMBER is defined in a Jenkins environment
if($ENV{BUILD_NUMBER} )
  string(CONCAT BUILD_VERSION ${VERSION_STRING} $ENV{BUILD_NUMBER})
else()
  string(CONCAT BUILD_VERSION ${VERSION_STRING} "0")
endif()

rocm_setup_version(VERSION ${BUILD_VERSION} NO_GIT_TAG_VERSION)
set(hiprand_VERSION "${rocrand_VERSION}")
math(EXPR temp "${rocrand_VERSION_MAJOR} * 100000 + ${rocrand_VERSION_MINOR} * 100 + ${rocrand_VERSION_PATCH}")
set(rocrand_VERSION_NUMBER ${temp})
set(hiprand_VERSION_NUMBER ${temp})

# AMD targets
set(AMDGPU_TARGETS gfx803;gfx900;gfx906 CACHE STRING "List of specific machine types for library to target")

# Tools
if (BUILD_TOOLS)
    add_subdirectory(tools)
endif()

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
