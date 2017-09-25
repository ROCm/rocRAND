# Package test
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(hiprand_package_test CXX C)

# Find HIP
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake/Modules)
find_package(HIP REQUIRED)
execute_process(
    COMMAND ${HIP_HIPCONFIG_EXECUTABLE} --cpp_config
    OUTPUT_VARIABLE HIP_CPP_CONFIG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
string(REPLACE " " ";" HIP_CPP_CONFIG ${HIP_CPP_CONFIG})
include(${PROJECT_SOURCE_DIR}/../../cmake/HIP.cmake)

# CUDA
if(HIP_PLATFORM STREQUAL "nvcc")
    set(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
    find_package(CUDA REQUIRED)
    set(HIP_HIPCC_FLAGS "-std=c++11")
endif()

# Find rocRAND
find_package(rocrand REQUIRED CONFIG HINTS ${rocrand_DIR} PATHS "/opt/rocm/rocrand")
# Find hipRAND
find_package(hiprand REQUIRED CONFIG HINTS ${hiprand_DIR} PATHS "/opt/rocm/hiprand")

# Check if Fortran wrapper is installed
if(NOT EXISTS "${hiprand_FORTRAN_SRC_DIRS}/hiprand_m.f90")
    message(FATAL_ERROR "${rocrand_FORTRAN_SRC_DIRS}/hiprand_m.f90 does not exist")
endif()
if(NOT EXISTS "${hiprand_FORTRAN_SRC_DIRS}/hip_m.f90")
    message(FATAL_ERROR "${hiprand_FORTRAN_SRC_DIRS}/hip_m.f90 does not exist")
endif()
if(NOT EXISTS "${rocrand_FORTRAN_SRC_DIRS}/rocrand_m.f90")
    message(FATAL_ERROR "${rocrand_FORTRAN_SRC_DIRS}/rocrand_m.f90 does not exist")
endif()
if(NOT EXISTS "${rocrand_FORTRAN_SRC_DIRS}/hip_m.f90")
    message(FATAL_ERROR "${rocrand_FORTRAN_SRC_DIRS}/hip_m.f90 does not exist")
endif()

# Get sources
file(GLOB rocrand_pkg_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand*.cpp)
file(GLOB hiprand_pkg_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_hiprand*.cpp)

# Enable testing (ctest)
enable_testing()

# Build
foreach(test_src ${rocrand_pkg_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_compile_options(${test_name} PRIVATE "${HIP_CPP_CONFIG}")
    if(HIP_PLATFORM STREQUAL "nvcc")
        target_link_libraries(${test_name}
            PRIVATE
                roc::rocrand
                ${CUDA_LIBRARIES}
        )
    else()
        target_link_libraries(${test_name}
            PRIVATE
                roc::rocrand
                "-L${HIP_ROOT_DIR}/lib -lhip_hcc -Wl,-rpath,${HIP_ROOT_DIR}/lib"
        )
    endif()
    add_test(NAME "${test_name}" COMMAND ${test_name})
endforeach()

foreach(test_src ${hiprand_pkg_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} "${test_src}")
    target_compile_options(${test_name} PRIVATE "${HIP_CPP_CONFIG}")
    if(HIP_PLATFORM STREQUAL "hcc")
        target_link_libraries(${test_name}
            PRIVATE
                hip::hiprand
                roc::rocrand
                "-L${HIP_ROOT_DIR}/lib -lhip_hcc -Wl,-rpath,${HIP_ROOT_DIR}/lib"
        )
    else()
        target_link_libraries(${test_name}
            PRIVATE
                hip::hiprand
                ${CUDA_curand_LIBRARY}
                ${CUDA_LIBRARIES}
        )
    endif()
    add_test(NAME "${test_name}" COMMAND ${test_name})
endforeach()
