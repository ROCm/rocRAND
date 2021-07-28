# MIT License
#
# Copyright (c) 2018-2021 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)

# Install prefix
if (WIN32)
  SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE )
  SET( INSTALL_PREFIX "C:/hipSDK" )
  SET( CPACK_SET_DESTDIR FALSE )
  SET( CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK" )
  SET( CPACK_PACKAGING_INSTALL_PREFIX "" )
  set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF )
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")

  #Adding CMAKE_PREFIX_PATH
  list( APPEND CMAKE_PREFIX_PATH /opt/rocm/llvm /opt/rocm )
endif()
# CMake modules
list(APPEND CMAKE_MODULE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules
    ${HIP_PATH}/cmake /opt/rocm/hip/cmake # FindHIP.cmake
    /opt/rocm/llvm
)

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

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOL "Add paths to linker search and installed rpath")

# Set the default value of BUILD_SHARED_LIBS
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared")

# Include cmake scripts
include(cmake/Common.cmake)
include(cmake/Dependencies.cmake)

# Detect compiler support for target ID
# This section is deprecated. Please use rocm_check_target_ids for future use.
if( CMAKE_CXX_COMPILER MATCHES ".*/hipcc$" )
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} "--help"
        OUTPUT_VARIABLE CXX_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE)
    string(REGEX MATCH ".mcode\-object\-version" TARGET_ID_SUPPORT ${CXX_OUTPUT})
endif()

set( AMDGPU_TARGETS "all" CACHE STRING "Compile for which gpu architectures?")
# Set the AMDGPU_TARGETS with backward compatiblity
if(COMMAND rocm_check_target_ids)
   rocm_check_target_ids(DEFAULT_AMDGPU_TARGETS
       TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx90a:xnack-;gfx90a:xnack+;gfx1030"
   )
else()
    # Use target ID syntax if supported for AMDGPU_TARGETS
    if(TARGET_ID_SUPPORT)
        set(DEFAULT_AMDGPU_TARGETS "gfx803;gfx900:xnack-;gfx906:xnack-;gfx908:xnack-;gfx1030")
    else()
        set(DEFAULT_AMDGPU_TARGETS "gfx803;gfx900;gfx906;gfx908")
    endif()
endif()
if (AMDGPU_TARGETS)
    if( AMDGPU_TARGETS STREQUAL "all" )
      set( gpus "${DEFAULT_AMDGPU_TARGETS}")
    else()
      set( gpus "${AMDGPU_TARGETS}")
    endif()
    # must FORCE set this AMDGPU_TARGETS before any find_package( hip ...), in this file
    # to override CACHE var and set --offload-arch flags via hip-config.cmake hip::device dependency
    set( AMDGPU_TARGETS "${gpus}" CACHE STRING "AMD GPU targets to compile for" FORCE )
endif()

#set(AMDGPU_TARGETS "${DEFAULT_AMDGPU_TARGETS}" CACHE STRING "List of specific machine types for library to target")

# Verify that hcc compiler is used on ROCM platform
# TODO: Fix VerifyCompiler for Windows
if (NOT WIN32)
  include(cmake/VerifyCompiler.cmake)
else()
  list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH} ${ROCM_PATH}/hip ${ROCM_PATH}/llvm /opt/rocm/llvm /opt/rocm/hip)
  find_package(hip REQUIRED CONFIG PATHS ${HIP_DIR} ${ROCM_PATH} /opt/rocm)
endif()

# Build option to disable -Werror
option(DISABLE_WERROR "Disable building with Werror" ON)

# Build CXX flags
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(DISABLE_WERROR)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -mf16c")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -mf16c")
endif()

# HIP on Windows: xhip is required with clang++ to get __half defined
if (WIN32)
  add_compile_options(-xhip)
endif()

# Build options
option(BUILD_FORTRAN_WRAPPER "Build Fortran wrapper" OFF)
option(BUILD_TEST "Build tests (requires googletest)" OFF)
option(BUILD_BENCHMARK "Build benchmarks" OFF)
option(BUILD_ADDRESS_SANITIZER "Build with address sanitizer enabled" OFF)

if(BUILD_ADDRESS_SANITIZER AND BUILD_SHARED_LIBS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -shared-libasan")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -shared-libasan")
  add_link_options(-fuse-ld=lld)
endif()

# Set version variables
rocm_setup_version( VERSION "2.10.9" )
set ( rocrand_VERSION ${rocRAND_VERSION} )
set ( hiprand_VERSION ${rocRAND_VERSION} )
# Old-style version number used within the library's API. rocrand_get_version should be modified.
math(EXPR temp "${rocRAND_VERSION_MAJOR} * 100000 + ${rocRAND_VERSION_MINOR} * 100 + ${rocRAND_VERSION_PATCH}")
set(hiprand_VERSION_NUMBER ${temp})
set(rocrand_VERSION_NUMBER ${temp})

# Print configuration summary
include(cmake/Summary.cmake)
print_configuration_summary()

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
