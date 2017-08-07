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
    find_package(Boost 1.54 QUIET COMPONENTS ${ROCRAND_BOOST_COMPONENTS})

    # Download, build, and install Boost if not found
    if(NOT Boost_FOUND)
        include(ExternalProject)
        include(FindPackageHandleStandardArgs)

        set(Boost_FIND_COMPONENTS ${ROCRAND_BOOST_COMPONENTS})
        set(BOOST_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/boost)

        if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(boost_am 64)
        else()
            set(boost_am 32)
        endif()
        mark_as_advanced(boost_am)

        set(boost_variant "release")
        if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
            set(boost_variant "debug")
        endif()
        mark_as_advanced(boost_variant)

        foreach(component ${Boost_FIND_COMPONENTS})
            list(APPEND Boost_COMPONENTS_FOR_BUILD --with-${component})
        endforeach()

        ExternalProject_Add(boost
            URL                 https://sourceforge.net/projects/boost/files/boost/1.64.0/boost_1_64_0.tar.bz2/download
            URL_MD5             93eecce2abed9d2442c9676914709349
            UPDATE_DISCONNECTED TRUE
            CONFIGURE_COMMAND   ./bootstrap.sh --prefix=${BOOST_ROOT_DIR}
            BUILD_COMMAND       ./b2 --prefix=${BOOST_ROOT_DIR} -d0 address-model=${boostlib_am} variant=${boost_variant} ${Boost_COMPONENTS_FOR_BUILD} install
            INSTALL_COMMAND     ""
            INSTALL_DIR         ${BOOST_ROOT_DIR}
            BUILD_IN_SOURCE     TRUE
        )
        ExternalProject_Get_Property(boost install_dir)
        set(BOOST_ROOT ${install_dir} CACHE PATH "")
        set(Boost_DIR ${install_dir} CACHE PATH "")
        set(Boost_INCLUDE_DIRS ${install_dir}/include)

        set(Boost_LIBRARIES)
        foreach(component ${Boost_FIND_COMPONENTS})
            list(APPEND Boost_LIBRARIES ${BOOST_ROOT_DIR}/lib/${LIBRARY_PREFIX}boost_${component}${LIBRARY_SUFFIX})
        endforeach()

        FIND_PACKAGE_HANDLE_STANDARD_ARGS(
            Boost DEFAULT_MSG
		    Boost_INCLUDE_DIRS Boost_LIBRARIES
		)
        mark_as_advanced(Boost_INCLUDE_DIRS Boost_LIBRARIES)

        # Add dependency
        list(APPEND ROCRAND_BENCHMARK_DEPENDENCIES boost)
        list(APPEND ROCRAND_CRUSH_TEST_DEPENDENCIES boost)
    endif()
endif()
