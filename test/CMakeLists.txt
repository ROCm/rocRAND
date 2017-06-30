# Tests

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(rocRAND_TEST_SRCS ${tmp})
set_source_files_properties(${rocRAND_TEST_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build
include_directories(${PROJECT_SOURCE_DIR}/library/include)
include_directories(SYSTEM ${googletest_SOURCE_DIR}/googletest/include)
foreach(test_src ${rocRAND_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    hip_add_executable(${test_name} "${test_src}")
    add_dependencies(${test_name} rocRAND gtest gtest_main)
    get_target_property(gtest_rpath gtest BINARY_DIR)
    get_target_property(rocRAND_rpath rocRAND BINARY_DIR)
    target_link_libraries(
        ${test_name}
        -Wl,-rpath,${gtest_rpath}:${rocRAND_rpath} # workaround for HIP linker
        $<TARGET_FILE:rocRAND> $<TARGET_FILE:gtest> $<TARGET_FILE:gtest_main>)
    set_target_properties(
        ${test_name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()
