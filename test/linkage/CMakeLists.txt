# Tests which check if there are no linkage problems when there
# are multiple sources (for example multiple definition errors).

# Add rocRAND linkage test
function(add_rocrand_link_test TEST_SOURCES)
    list(GET TEST_SOURCES 0 TEST_MAIN_SOURCE)
    get_filename_component(TEST_TARGET ${TEST_MAIN_SOURCE} NAME_WE)
    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    # nvcc/CUDA
    if(HIP_COMPILER STREQUAL "nvcc")
        set_source_files_properties(${TEST_SOURCES}
          PROPERTIES
            LANGUAGE CUDA
            CUDA_STANDARD 14
        )
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
    if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
        target_link_libraries(${TEST_TARGET} hip::device)
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
    add_executable(${TEST_TARGET} ${TEST_SOURCES})
    # nvcc/CUDA
    if(HIP_COMPILER STREQUAL "nvcc")
      set_source_files_properties(${TEST_SOURCES}
        PROPERTIES
          LANGUAGE CUDA
          CUDA_STANDARD 14
      )
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
    if(HIP_COMPILER STREQUAL "nvcc")
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
        target_link_libraries(${TEST_TARGET} hip::device)
    endif()
    set_target_properties(${TEST_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${TEST_TARGET} COMMAND $<TARGET_FILE:${TEST_TARGET}>)
endfunction()

add_hiprand_link_test("test_hiprand_linkage.cpp;get_hiprand_version.cpp")
