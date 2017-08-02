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

if(BUILD_BENCHMARK OR BUILD_TEST)
    file(GLOB tmp ${PROJECT_SOURCE_DIR}/cmake/Modules/program_options/src/*.cpp)
    include_directories(${PROJECT_SOURCE_DIR}/cmake/Modules/program_options/include)
    add_library(program_options SHARED "${tmp}")
    set_target_properties(
    program_options
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
)
endif()
