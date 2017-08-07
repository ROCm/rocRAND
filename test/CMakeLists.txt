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
include_directories(${PROJECT_SOURCE_DIR}/library/include)
include_directories(${PROJECT_SOURCE_DIR}/library/src)
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

# Crush Tests
if(BUILD_CRUSH_TEST)
    # Download TestU01 library
    download_project(PROJ                TestU01
                     GIT_REPOSITORY      https://github.com/JamesHirschorn/TestU01-CMake.git
                     GIT_TAG             master
                     UPDATE_DISCONNECTED TRUE
    )
    add_subdirectory(${TestU01_SOURCE_DIR} ${TestU01_BINARY_DIR})
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
        add_dependencies(${test_name} rocRAND)
    endif()
    add_dependencies(${test_name} hipRAND gtest gtest_main)
    get_target_property(gtest_rpath gtest BINARY_DIR)
    get_target_property(hipRAND_rpath hipRAND BINARY_DIR)
    if(HIP_PLATFORM STREQUAL "hcc")
        get_target_property(rocRAND_rpath rocRAND BINARY_DIR)
        target_link_libraries(
            ${test_name}
            -Wl,-rpath,${gtest_rpath}:${hipRAND_rpath}:${rocRAND_rpath} # workaround for HIP linker
            $<TARGET_FILE:hipRAND> $<TARGET_FILE:rocRAND>
            $<TARGET_FILE:gtest> $<TARGET_FILE:gtest_main>
        )
    else()
        target_link_libraries(
            ${test_name}
            "-lcurand" hipRAND
            gtest gtest_main
        )
    endif()
    set_target_properties(
        ${test_name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()