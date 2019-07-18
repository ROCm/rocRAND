# Tests which check if there are no linkage problems when there
# are multiple sources (for example multiple definition errors).

# Use CUDA_INCLUDE_DIRECTORIES to include required dirs
# for nvcc if cmake version is less than 3.9.3
if((HIP_PLATFORM STREQUAL "nvcc") AND (CMAKE_VERSION VERSION_LESS "3.9.3"))
    CUDA_INCLUDE_DIRECTORIES(
        "${PROJECT_BINARY_DIR}/library/include/"
        "${PROJECT_SOURCE_DIR}/library/include/"
        "${PROJECT_SOURCE_DIR}/library/src"
        ${GTEST_INCLUDE_DIRS}
    )
endif()

# Add rocRAND linkage test
function(add_rocrand_link_test TEST_SOURCES)
    list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
    get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
    # nvcc/CUDA
    if(HIP_PLATFORM STREQUAL "nvcc")
        set_source_files_properties(${TEST_SOURCES}
            PROPERTIES
                CUDA_SOURCE_PROPERTY_FORMAT OBJ
        )
        CUDA_ADD_EXECUTABLE(${TEST_TARGET} ${TEST_SOURCES})
    # hcc/ROCm
    else()
        add_executable(${TEST_TARGET} ${TEST_SOURCES})
    endif()
    # GTEST include dirs
    target_include_directories(${TEST_TARGET} SYSTEM
        PUBLIC
            ${GTEST_INCLUDE_DIRS}
    )
    target_include_directories(${TEST_TARGET}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/library/src
    )
    target_link_libraries(${TEST_TARGET}
        rocrand
        ${GTEST_BOTH_LIBRARIES}
    )
    if(HIP_PLATFORM STREQUAL "hcc")
        # Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
        if(TARGET hip::device)
            target_link_libraries(${TEST_TARGET} hip::device)
        else()
            target_link_libraries(${TEST_TARGET} hip::hip_hcc hip::hip_device)
        endif()
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${TEST_TARGET} --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(
        ${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endfunction()

add_rocrand_link_test("test_rocrand_linkage.cpp;get_rocrand_version.cpp")

# Add hipRAND linkage test
function(add_hiprand_link_test TEST_SOURCES)
    list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
    get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
    # nvcc/CUDA
    if(HIP_PLATFORM STREQUAL "nvcc")
        set_source_files_properties(${TEST_SOURCES}
            PROPERTIES
                CUDA_SOURCE_PROPERTY_FORMAT OBJ
        )
        CUDA_ADD_EXECUTABLE(${TEST_TARGET} ${TEST_SOURCES})
    # hcc/ROCm
    else()
        add_executable(${TEST_TARGET} ${TEST_SOURCES})
    endif()
    # GTEST include dirs
    target_include_directories(${TEST_TARGET} SYSTEM
        PUBLIC
            ${GTEST_INCLUDE_DIRS}
    )
    target_include_directories(${TEST_TARGET}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/library/src
    )
    if(HIP_PLATFORM STREQUAL "nvcc")
        target_link_libraries(${TEST_TARGET}
            hiprand
            ${CUDA_curand_LIBRARY}
            ${GTEST_BOTH_LIBRARIES}
        )
    else()
        target_link_libraries(${TEST_TARGET}
            hiprand
            rocrand
            ${GTEST_BOTH_LIBRARIES}
        )
        # Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
        if(TARGET hip::device)
            target_link_libraries(${TEST_TARGET} hip::device)
        else()
            target_link_libraries(${TEST_TARGET} hip::hip_hcc hip::hip_device)
        endif()
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${TEST_TARGET} --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endfunction()

add_hiprand_link_test("test_hiprand_linkage.cpp;get_hiprand_version.cpp")
