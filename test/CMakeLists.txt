# Tests
option(BUILD_CRUSH_TEST "Build Crush tests (requires testu01)" OFF)

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(rocRAND_TEST_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/test_hiprand*.cpp)
foreach(EXCLUDE ${tmp})
    list(REMOVE_ITEM rocRAND_TEST_SRCS "${EXCLUDE}")
endforeach()
set_source_files_properties(${rocRAND_TEST_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build
include_directories(${PROJECT_SOURCE_DIR}/library/include) # hipcc workaround
include_directories(${PROJECT_BINARY_DIR}/library/include) # hipcc workaround
include_directories(${PROJECT_SOURCE_DIR}/library/src) # hipcc workaround
include_directories(SYSTEM ${GTEST_INCLUDE_DIRS}) # hipcc workaround

foreach(test_src ${rocRAND_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    hip_add_executable(${test_name} "${test_src}")
    add_dependencies(${test_name} rocrand ${ROCRAND_TEST_DEPENDENCIES})
    get_target_property(rocRAND_rpath rocrand BINARY_DIR)
    if(NOT GTEST_ROOT)
        set(GTest_rpath "${PROJECT_BINARY_DIR}/gtest/lib")
    else()
        set(GTest_rpath "${GTEST_ROOT}/lib")
    endif()
    target_link_libraries(
        ${test_name}
        -Wl,-rpath,${rocRAND_rpath}:${GTest_rpath} # workaround for HIP linker
        $<TARGET_FILE:rocrand>
        ${GTEST_BOTH_LIBRARIES}
    )
    if(HIP_PLATFORM STREQUAL "hcc")
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${test_name} --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(
        ${test_name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()

# Fortran Wrapper Tests
if (BUILD_FORTRAN_WRAPPER)
    add_subdirectory(fortran)
endif()

# Crush Tests
if(BUILD_CRUSH_TEST)
    add_subdirectory(crush)
endif()

# hipRand Tests
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/test_hiprand*.cpp)
set(hipRAND_TEST_SRCS ${tmp})
set_source_files_properties(${hipRAND_TEST_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

foreach(test_src ${hipRAND_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    hip_add_executable(${test_name} "${test_src}")
    if(HIP_PLATFORM STREQUAL "hcc")
        add_dependencies(${test_name} rocrand)
    endif()
    add_dependencies(${test_name} hiprand ${ROCRAND_TEST_DEPENDENCIES})
    get_target_property(hipRAND_rpath hiprand BINARY_DIR)
    if(HIP_PLATFORM STREQUAL "hcc")
        get_target_property(rocRAND_rpath rocrand BINARY_DIR)
        if(NOT GTEST_ROOT)
            set(GTest_rpath "${PROJECT_BINARY_DIR}/gtest/lib")
        else()
            set(GTest_rpath "${GTEST_ROOT}/lib")
        endif()
        target_link_libraries(
            ${test_name}
            -Wl,-rpath,${hipRAND_rpath}:${rocRAND_rpath}:${GTest_rpath} # workaround for HIP linker
            $<TARGET_FILE:hiprand> $<TARGET_FILE:rocrand>
            ${GTEST_BOTH_LIBRARIES}
        )
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${test_name} --amdgpu-target=${amdgpu_target})
        endforeach()
    else()
        target_link_libraries(
            ${test_name}
            "-lcurand" hiprand
            ${GTEST_BOTH_LIBRARIES}
        )
    endif()
    set_target_properties(
        ${test_name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()
