# Dependencies

# GIT
find_package(Git REQUIRED)
if (NOT Git_FOUND)
    message(FATAL_ERROR "Please ensure Git is installed on the system")
endif()

# HIP
find_package(HIP REQUIRED)
include_directories(SYSTEM ${HIP_INCLUDE_DIRECTORIES})
include(cmake/HIP.cmake)

set(ROCRAND_TEST_DEPENDENCIES)
set(ROCRAND_CRUSH_TEST_DEPENDENCIES)
set(ROCRAND_BENCHMARK_DEPENDENCIES)

# Test dependencies
if (BUILD_TEST)
    include(cmake/DownloadProject.cmake)
    # Download googletest library
    download_project(PROJ                googletest
                     GIT_REPOSITORY      https://github.com/google/googletest.git
                     GIT_TAG             master
                     UPDATE_DISCONNECTED TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
    list(APPEND ROCRAND_TEST_DEPENDENCIES gtest gtest_main)
endif()

# Crush Tests
if(BUILD_CRUSH_TEST)
    # Download TestU01 library
    download_project(PROJ                TestU01
                     GIT_REPOSITORY      https://github.com/JamesHirschorn/TestU01-CMake.git
                     GIT_TAG             master
                     UPDATE_DISCONNECTED TRUE
    )
    add_subdirectory(${TestU01_SOURCE_DIR} ${TestU01_BINARY_DIR})
    # Add dependency
    list(APPEND ROCRAND_CRUSH_TEST_DEPENDENCIES TestU01)
endif()

# Boost.Program_options is required only for benchmarks and crush tests
if(BUILD_BENCHMARK OR BUILD_CRUSH_TEST)
    set(ROCRAND_BOOST_COMPONENTS program_options)
    if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
        find_package(Boost 1.54 QUIET COMPONENTS ${ROCRAND_BOOST_COMPONENTS})
    endif()

    # Download, build, and install Boost if not found
    if(NOT Boost_FOUND)
        message(STATUS "Boost not found. Downloading and building Boost.")
        set(BOOST_ROOT ${CMAKE_BINARY_DIR}/boost CACHE PATH "")
        configure_file(
            ${CMAKE_CURRENT_LIST_DIR}/DownloadBoost.CMakeLists.txt.in boost-download/CMakeLists.txt
            @ONLY
        )
        # Configure
        execute_process(
            COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" -D "CMAKE_MAKE_PROGRAM:FILE=${CMAKE_MAKE_PROGRAM}" .
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/boost-download
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "CMake/config step for Boost failed: ${result}")
            unset(BOOST_ROOT)
        endif()
        # Build and install
        execute_process(
            COMMAND ${CMAKE_COMMAND} --build .
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/boost-download
            RESULT_VARIABLE result
        )
        if(result)
            message(FATAL_ERROR "Build step for Boost failed: ${result}")
            unset(BOOST_ROOT)
        endif()

        find_package(Boost 1.54
            REQUIRED
            COMPONENTS ${ROCRAND_BOOST_COMPONENTS}
        )
    endif()
endif()
