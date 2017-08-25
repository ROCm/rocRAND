# Package test
cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)

project(hiprand_package_test CXX C)

# Find HIP
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../../cmake/Modules)
find_package(HIP REQUIRED)
include_directories(SYSTEM ${HIP_INCLUDE_DIRECTORIES})
include(${PROJECT_SOURCE_DIR}/../../cmake/HIP.cmake)

# Find rocRAND
find_package(rocrand REQUIRED CONFIG HINTS ${ROCRAND_DIR} PATHS "/opt/rocm/rocrand")
include_directories(${rocrand_INCLUDE_DIRS})

# Find hipRAND
find_package(hiprand REQUIRED CONFIG HINTS ${HIPRAND_DIR} PATHS "/opt/rocm/hiprand")
include_directories(${hiprand_INCLUDE_DIRS})

# Get sources
file(GLOB rocrand_pkg_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_rocrand*.cpp)
set_source_files_properties(${rocrand_pkg_TEST_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)
file(GLOB hiprand_pkg_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_hiprand*.cpp)
set_source_files_properties(${hiprand_pkg_TEST_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Enable testing (ctest)
enable_testing()

# Build
foreach(test_src ${rocrand_pkg_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    hip_add_executable(${test_name} "${test_src}")
    target_link_libraries(${test_name} ${rocrand_LIBRARIES})
    add_test(NAME "${test_name}" COMMAND ${test_name})
endforeach()
foreach(test_src ${hiprand_pkg_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    hip_add_executable(${test_name} "${test_src}")
    if(HIP_PLATFORM STREQUAL "hcc")
        target_link_libraries(${test_name} ${rocrand_LIBRARIES} ${hiprand_LIBRARIES})
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${test_name} --amdgpu-target=${amdgpu_target})
        endforeach()
    else()
        target_link_libraries(${test_name} "-lcurand" ${hiprand_LIBRARIES})
    endif()
    add_test(NAME "${test_name}" COMMAND ${test_name})
endforeach()
