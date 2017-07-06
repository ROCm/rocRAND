# Dependencies

# HIP
find_package(HIP REQUIRED)
include_directories(SYSTEM ${HIP_INCLUDE_DIRECTORIES})
include(cmake/HIP.cmake)

# Test dependencies
if (BUILD_TEST)
    include(cmake/DownloadProject.cmake)

    # Download googletest library
    download_project(PROJ                googletest
                     GIT_REPOSITORY      https://github.com/google/googletest.git
                     GIT_TAG             master
                     ${UPDATE_DISCONNECTED_IF_AVAILABLE}
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})
endif()

if(BUILD_BENCHMARK)
    set(BENCHMARK_BOOST_COMPONENTS program_options)
    find_package(Boost 1.54 REQUIRED COMPONENTS ${BENCHMARK_BOOST_COMPONENTS})
endif()
