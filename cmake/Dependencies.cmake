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

# For downloading, building, and installing required dependencies
include(cmake/DownloadProject.cmake)

if (CMAKE_VERSION VERSION_LESS 3.2)
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "")
else()
    set(UPDATE_DISCONNECTED_IF_AVAILABLE "UPDATE_DISCONNECTED TRUE")
endif()

# Test dependencies
if(BUILD_TEST)
    if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
        find_package(GTest QUIET)
    endif()

    if(NOT GTEST_FOUND)
        message(STATUS "GTest not found. Downloading and building GTest.")
        # Download, build and install googletest library
        set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
        download_project(PROJ                googletest
                         GIT_REPOSITORY      https://github.com/google/googletest.git
                         GIT_TAG             master
                         INSTALL_DIR         ${GTEST_ROOT}
                         CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                         LOG_DOWNLOAD        TRUE
                         LOG_CONFIGURE       TRUE
                         LOG_BUILD           TRUE
                         LOG_INSTALL         TRUE
                         ${UPDATE_DISCONNECTED_IF_AVAILABLE}
        )
    endif()
    find_package(GTest REQUIRED)
endif()

# Crush Tests
if(BUILD_CRUSH_TEST)
    if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
        find_package(TestU01 QUIET)
    endif()

    if(NOT TestU01_FOUND)
        message(STATUS "TestU01 not found. Downloading and building TestU01.")
        # Download and install TestU01 library
        set(TESTU01_ROOT_DIR ${CMAKE_CURRENT_BINARY_DIR}/testu01)
        download_project(PROJ                TestU01
                         URL                 http://simul.iro.umontreal.ca/testu01/TestU01.zip
                         URL_MD5             0cbbe837f330d813ee258ef6bea2ac0e
                         INSTALL_DIR         ${TESTU01_ROOT_DIR}
                         CONFIGURE_COMMAND   ./configure --prefix=<INSTALL_DIR>
                         BUILD_COMMAND       make
                         INSTALL_COMMAND     make install
                         UPDATE_COMMAND      ""
                         PATCH_COMMAND       ""
                         LOG_DOWNLOAD        TRUE
                         LOG_CONFIGURE       TRUE
                         LOG_BUILD           TRUE
                         LOG_INSTALL         TRUE
        )
    endif()
    find_package(TestU01 REQUIRED)
endif()
