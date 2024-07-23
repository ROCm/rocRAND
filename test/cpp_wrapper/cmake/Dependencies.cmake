# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

cmake_minimum_required(VERSION 3.16)

# find_package() uses upper-case <PACKAGENAME>_ROOT variables.
# altough we use GTEST_ROOT for our purposes, it is actually even benefecial for
# find_package() to look for it there (that's where we are going to put it anyway)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# Dependencies

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# For downloading, building, and installing required dependencies
include(../../cmake/DownloadProject.cmake)

# NOTE: Google Test has created a mess with legacy FindGTest.cmake and newer GTestConfig.cmake
#
# FindGTest.cmake defines:   GTest::GTest, GTest::Main, GTEST_FOUND
#
# GTestConfig.cmake defines: GTest::gtest, GTest::gtest_main, GTest::gmock, GTest::gmock_main
#
# NOTE2: Finding GTest in MODULE mode, one cannot invoke find_package in CONFIG mode, because targets
#        will be duplicately defined.
if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
  # Google Test (https://github.com/google/googletest)
  find_package(GTest QUIET)
endif()

if(NOT TARGET GTest::GTest AND NOT TARGET GTest::gtest)
  message(STATUS "GTest not found or force download GTest on. Downloading and building GTest.")
  set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/gtest CACHE PATH "")
  if(DEFINED CMAKE_CXX_COMPILER)
    set(CXX_COMPILER_OPTION "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
  endif()
  download_project(
    PROJ                googletest
    GIT_REPOSITORY      https://github.com/google/googletest.git
    GIT_TAG             release-1.11.0
    INSTALL_DIR         ${GTEST_ROOT}
    CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> ${CXX_COMPILER_OPTION} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    LOG_DOWNLOAD        TRUE
    LOG_CONFIGURE       TRUE
    LOG_BUILD           TRUE
    LOG_INSTALL         TRUE
    BUILD_PROJECT       TRUE
    UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
  )
  find_package(GTest CONFIG REQUIRED PATHS ${GTEST_ROOT} NO_DEFAULT_PATH)
endif()
