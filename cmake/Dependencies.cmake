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

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

if(USE_HIP_CPU)
  find_package(Threads REQUIRED)

  set(CMAKE_REQUIRED_FLAGS "-std=c++17")
  include(CheckCXXSymbolExists)
  check_cxx_symbol_exists(__GLIBCXX__ "cstddef" STL_IS_GLIBCXX)
  set(STL_DEPENDS_ON_TBB ${STL_IS_GLIBCXX})
  if(STL_DEPENDS_ON_TBB)
    if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
      # TBB (https://github.com/oneapi-src/oneTBB)
      find_package(TBB QUIET)
    endif()

    if(NOT TBB_FOUND)
      message(STATUS "TBB not found or force download TBB on. Downloading and building TBB.")
      if(CMAKE_CONFIGURATION_TYPES)
        message(FATAL_ERROR "DownloadProject.cmake doesn't support multi-configuration generators.")
      endif()
      set(TBB_ROOT ${CMAKE_CURRENT_BINARY_DIR}/deps/tbb CACHE PATH "")
      download_project(
        PROJ                tbb
        GIT_REPOSITORY      https://github.com/oneapi-src/oneTBB.git
        GIT_TAG             v2020.3
        CONFIGURE_COMMAND "${CMAKE_COMMAND} -E true"
        BUILD_COMMAND "${CMAKE_COMMAND} -E true"
        INSTALL_COMMAND "${CMAKE_COMMAND} -E true"
        UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
      )
      find_program(CLANG_EXE
        clang
        PATH_SUFFIXES bin llvm/bin
        REQUIRED
      )
      get_filename_component(CLANG_DIR "${CLANG_EXE}" DIRECTORY)
      set(TBB_SOURCE_DIR "${CMAKE_BINARY_DIR}/tbb-src")
      list(APPEND CMAKE_MODULE_PATH "${TBB_SOURCE_DIR}/cmake")
      include(TBBBuild)
      set(ENV{PATH} "$ENV{PATH}:${CLANG_DIR}")
      tbb_build(TBB_ROOT "${TBB_SOURCE_DIR}" CONFIG_DIR TBB_CONFIG_DIR MAKE_ARGS tbb_build_dir=${TBB_ROOT})
    endif()
    find_package(TBB REQUIRED CONFIG PATHS ${TBB_CONFIG_DIR} NO_DEFAULT_PATH)
  endif(STL_DEPENDS_ON_TBB)

  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    # HIP CPU Runtime (https://github.com/ROCm-Developer-Tools/HIP-CPU)
    find_package(hip_cpu_rt QUIET)
  endif()

  if(NOT hip_cpu_rt_FOUND)
    message(STATUS "Downloading and building HIP CPU Runtime.")
    set(HIP_CPU_ROOT "${CMAKE_CURRENT_BINARY_DIR}/deps/hip-cpu" CACHE PATH "")
    download_project(
      PROJ                hip-cpu
      GIT_REPOSITORY      https://github.com/ROCm-Developer-Tools/HIP-CPU.git
      GIT_TAG             master
      INSTALL_DIR         "${HIP_CPU_ROOT}"
      CMAKE_ARGS          -Dhip_cpu_rt_BUILD_EXAMPLES=OFF -Dhip_cpu_rt_BUILD_TESTING=OFF -DTBB_DIR=${TBB_CONFIG_DIR} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
  endif()
  find_package(hip_cpu_rt REQUIRED CONFIG PATHS ${HIP_CPU_ROOT})
endif()

# Fortran Wrapper
if(BUILD_FORTRAN_WRAPPER)
    enable_language(Fortran)
endif()

# Test dependencies
if(BUILD_TEST)
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
    download_project(
      PROJ                googletest
      GIT_REPOSITORY      https://github.com/google/googletest.git
      GIT_TAG             release-1.11.0
      INSTALL_DIR         ${GTEST_ROOT}
      CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      LOG_DOWNLOAD        TRUE
      LOG_CONFIGURE       TRUE
      LOG_BUILD           TRUE
      LOG_INSTALL         TRUE
      BUILD_PROJECT       TRUE
      UPDATE_DISCONNECTED TRUE # Never update automatically from the remote repository
    )
    list( APPEND CMAKE_PREFIX_PATH ${GTEST_ROOT} )
    find_package(GTest CONFIG REQUIRED PATHS ${GTEST_ROOT})
  endif()
endif()

# Find or download/install rocm-cmake project
find_package(ROCM 0.7.3 QUIET CONFIG PATHS $ENV{ROCM_PATH})
if(NOT ROCM_FOUND)
  set(PROJECT_EXTERN_DIR "${CMAKE_CURRENT_BINARY_DIR}/deps")
  file( TO_NATIVE_PATH "${PROJECT_EXTERN_DIR}" PROJECT_EXTERN_DIR_NATIVE)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(
      DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.tar.gz
      ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.tar.gz
      STATUS rocm_cmake_download_status LOG rocm_cmake_download_log
  )
  list(GET rocm_cmake_download_status 0 rocm_cmake_download_error_code)
  if(rocm_cmake_download_error_code)
      message(FATAL_ERROR "Error: downloading "
          "https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip failed "
          "error_code: ${rocm_cmake_download_error_code} "
          "log: ${rocm_cmake_download_log} "
      )
  endif()

  execute_process(
      COMMAND ${CMAKE_COMMAND} -E tar xzvf ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}.tar.gz
      WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}
  )
  execute_process(
      COMMAND ${CMAKE_COMMAND} -S ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag} -B ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}/build
      WORKING_DIRECTORY ${PROJECT_EXTERN_DIR}
  )
  execute_process(
      COMMAND ${CMAKE_COMMAND} --install ${PROJECT_EXTERN_DIR}/rocm-cmake-${rocm_cmake_tag}/build --prefix ${PROJECT_EXTERN_DIR}/rocm
      WORKING_DIRECTORY ${PROJECT_EXTERN_DIR} )
  if(rocm_cmake_unpack_error_code)
      message(FATAL_ERROR "Error: unpacking ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip failed")
  endif()
  find_package(ROCM 0.7.3 REQUIRED CONFIG PATHS ${PROJECT_EXTERN_DIR})
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
include(ROCMCheckTargetIds)
include(ROCMUtilities)
include(ROCMClients)
include(ROCMHeaderWrapper)
