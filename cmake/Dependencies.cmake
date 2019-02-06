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

# Dependencies

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# GIT
find_package(Git REQUIRED)
if(NOT Git_FOUND)
    message(FATAL_ERROR "Please ensure Git is installed on the system")
endif()

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

if(CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED TRUE")
endif()

# Fortran Wrapper
if(BUILD_FORTRAN_WRAPPER)
    enable_language(Fortran)
endif()

# Test dependencies
if(BUILD_TEST)
    if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
        find_package(GTest QUIET)
    endif()

    if(NOT GTEST_FOUND)
        message(STATUS "GTest not found. Downloading and building GTest.")
        # Download, build and install googletest library
        set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
        download_project(PROJ                googletest
                         GIT_REPOSITORY      https://github.com/google/googletest.git
                         GIT_TAG             release-1.8.1
                         INSTALL_DIR         ${GTEST_ROOT}
                         CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                         LOG_DOWNLOAD        TRUE
                         LOG_CONFIGURE       TRUE
                         LOG_BUILD           TRUE
                         LOG_INSTALL         TRUE
                         ${UPDATE_DISCONNECTED_IF_AVAILABLE}
        )
    endif()
    find_package(GTest REQUIRED)
endif()

# Crush Tests
if(BUILD_CRUSH_TEST)
    if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
        find_package(TestU01 QUIET)
    endif()

    if(NOT TestU01_FOUND)
        message(STATUS "TestU01 not found. Downloading and building TestU01.")
        # Download and install TestU01 library
        set(TESTU01_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/testu01)
        download_project(PROJ                TestU01
                         URL                 http://simul.iro.umontreal.ca/testu01/TestU01.zip
                         URL_MD5             0cbbe837f330d813ee258ef6bea2ac0e
                         INSTALL_DIR         ${TESTU01_ROOT_DIR}
                         CONFIGURE_COMMAND   ./configure --prefix=<INSTALL_DIR>
                         BUILD_COMMAND       make
                         INSTALL_COMMAND     make install
                         UPDATE_COMMAND      ""
                         PATCH_COMMAND       ""
                         LOG_DOWNLOAD        TRUE
                         LOG_CONFIGURE       TRUE
                         LOG_BUILD           TRUE
                         LOG_INSTALL         TRUE
        )
    endif()
    find_package(TestU01 REQUIRED)
endif()
