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
endif()

# Boost.Program_options is required only for benchmarks and crush tests
if(BUILD_BENCHMARK OR BUILD_CRUSH_TEST)
    # TODO: Download, build, and install Boost if not found
    set(ROCRAND_BOOST_COMPONENTS program_options)
    find_package(Boost 1.54 REQUIRED COMPONENTS ${ROCRAND_BOOST_COMPONENTS})
endif()
