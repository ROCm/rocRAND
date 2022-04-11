# Package test
cmake_minimum_required(VERSION 3.5.1 FATAL_ERROR)

project(rocrand_package_test CXX)

# CMake modules
list(APPEND CMAKE_PREFIX_PATH $ENV{ROCM_PATH} $ENV{ROCM_PATH}/hip)
list(APPEND CMAKE_MODULE_PATH
    ${HIP_PATH}/cmake $ENV{ROCM_PATH}/hip/cmake # FindHIP.cmake
)

# Find HIP
if(CMAKE_CXX_COMPILER MATCHES ".*/nvcc$" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    find_package(hip QUIET CONFIG PATHS $ENV{ROCM_PATH})
    if(NOT hip_FOUND)
        find_package(HIP REQUIRED)
        if((HIP_COMPILER STREQUAL "hcc") AND (HIP_PLATFORM STREQUAL "nvcc"))
           # TODO: The HIP package on NVIDIA platform is incorrect at few versions
           set(HIP_COMPILER "nvcc" CACHE STRING "HIP Compiler" FORCE)
        endif()
    endif()
else()
  find_package(hip REQUIRED CONFIG PATHS $ENV{ROCM_PATH})
endif()

# Get HIP options
execute_process(
    COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --cpp_config
    OUTPUT_VARIABLE HIP_CPP_CONFIG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${HIP_CPP_CONFIG}")

# CUDA
if(HIP_COMPILER STREQUAL "nvcc")
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    find_package(CUDA REQUIRED)
endif()

# Find rocRAND
find_package(rocrand REQUIRED CONFIG HINTS ${rocrand_DIR} PATHS "$ENV{ROCM_PATH}/rocrand")

# TODO: Fix fortran wrapper
# Check if Fortran wrapper is installed
#if(NOT EXISTS "${hiprand_FORTRAN_SRC_DIRS}/hiprand_m.f90")
#    message(FATAL_ERROR "${rocrand_FORTRAN_SRC_DIRS}/hiprand_m.f90 does not exist")
#endif()
#if(NOT EXISTS "${hiprand_FORTRAN_SRC_DIRS}/hip_m.f90")
#    message(FATAL_ERROR "${hiprand_FORTRAN_SRC_DIRS}/hip_m.f90 does not exist")
#endif()
#if(NOT EXISTS "${rocrand_FORTRAN_SRC_DIRS}/rocrand_m.f90")
#    message(FATAL_ERROR "${rocrand_FORTRAN_SRC_DIRS}/rocrand_m.f90 does not exist")
#endif()
#if(NOT EXISTS "${rocrand_FORTRAN_SRC_DIRS}/hip_m.f90")
#    message(FATAL_ERROR "${rocrand_FORTRAN_SRC_DIRS}/hip_m.f90 does not exist")
#endif()

# Get sources
file(GLOB rocrand_pkg_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand*.cpp)

# Enable testing (ctest)
enable_testing()

function(add_relative_test test_name test_target)
    get_target_property(EXE_PATH ${test_target} RUNTIME_OUTPUT_DIRECTORY)
    if(EXE_PATH STREQUAL "EXE_PATH-NOTFOUND")
        set(EXE_PATH ".")
    endif()
    get_filename_component(EXE_PATH "${EXE_PATH}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    get_target_property(EXE_NAME ${test_target} RUNTIME_OUTPUT_NAME)
    if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
        get_target_property(EXE_NAME ${test_target} OUTPUT_NAME)
        if(EXE_NAME STREQUAL "EXE_NAME-NOTFOUND")
            set(EXE_NAME "${test_target}")
        endif()
    endif()
    file(RELATIVE_PATH rel_path "${CMAKE_CURRENT_BINARY_DIR}" "${EXE_PATH}/${EXE_NAME}")
    add_test(NAME "${test_name}" COMMAND "./${rel_path}")
endfunction()

# Build
foreach(test_src ${rocrand_pkg_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    if(HIP_COMPILER STREQUAL "nvcc")
        target_link_libraries(${test_name}
            PRIVATE
                roc::rocrand
                ${CUDA_LIBRARIES}
        )
    else()
        target_link_libraries(${test_name}
            PRIVATE
                roc::rocrand
                "-L${HIP_ROOT_DIR}/lib -Wl,-rpath,${HIP_ROOT_DIR}/lib"
        )
    endif()
    add_relative_test(${test_name} ${test_name})
    rocm_install(TARGETS ${test_name} COMPONENT tests)
    if (WIN32 AND NOT DEFINED DLLS_COPIED)
      set(DLLS_COPIED "YES")
      set(DLLS_COPIED ${DLLS_COPIED} PARENT_SCOPE)
      # for now adding in all .dll as dependency chain is not cmake based on win32
      file( GLOB third_party_dlls
      LIST_DIRECTORIES ON
      CONFIGURE_DEPENDS
      ${HIP_DIR}/bin/*.dll
      ${CMAKE_SOURCE_DIR}/rtest.*
      )
      foreach( file_i ${third_party_dlls})
        add_custom_command( TARGET ${test_name} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/test/package )
      endforeach( file_i )
    endif() 
endforeach()
