# rocRAND library

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(rocRAND_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/*.cpp)
foreach(EXCLUDE ${tmp})
    list(REMOVE_ITEM rocRAND_SRCS "${EXCLUDE}")
endforeach()
set_source_files_properties(${rocRAND_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# configure a header file to pass the rocRAND version
configure_file(
    "${PROJECT_SOURCE_DIR}/library/include/rocrand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/rocrand_version.h"
    @ONLY
)

include_directories(include) # hipcc workaround
include_directories(${PROJECT_BINARY_DIR}/library/include) # hipcc workaround

# Build library
hip_add_library(rocrand SHARED "${rocRAND_SRCS}")
target_include_directories(rocrand
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
        $<INSTALL_INTERFACE:rocrand/include>
)
set_target_properties(rocrand
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
        DEBUG_POSTFIX "-d"
        VERSION ${rocrand_VERSION}
        SOVERSION 1
)
if(HIP_PLATFORM STREQUAL "hcc")
    foreach(amdgpu_target ${AMDGPU_TARGETS})
        target_link_libraries(rocrand PRIVATE --amdgpu-target=${amdgpu_target})
    endforeach()
endif()

# Install
# .so lib
install(
    TARGETS rocrand
    EXPORT rocrand-targets
    LIBRARY DESTINATION rocrand/lib
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)
# headers
install(
    DIRECTORY
        "include/"
        "${PROJECT_BINARY_DIR}/library/include/"
    DESTINATION rocrand/include
    FILES_MATCHING
    PATTERN "rocrand*h"
    PATTERN "rocrand*hpp"
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
# .cmake files
install(
    EXPORT rocrand-targets
    NAMESPACE roc::
    DESTINATION rocrand/lib/cmake/rocrand
)
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/rocrand/include")
set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/rocrand/lib")
include(CMakePackageConfigHelpers)
configure_package_config_file(
    src/rocrand-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/rocrand-config.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/rocrand
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rocrand-config-version.cmake
    VERSION ${rocrand_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/rocrand-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/rocrand-config-version.cmake
    DESTINATION rocrand/lib/cmake/rocrand
)

# hipRAND library

# Get sources
file(GLOB hipRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/*_${HIP_PLATFORM}.cpp)
set_source_files_properties(${hipRAND_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# configure a header file to pass the hipRAND version
configure_file(
    "${PROJECT_SOURCE_DIR}/library/include/hiprand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/hiprand_version.h"
    @ONLY
)

# Build
hip_add_library(hiprand SHARED "${hipRAND_SRCS}")
target_include_directories(hiprand
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
        $<INSTALL_INTERFACE:hiprand/include>
)
if(HIP_PLATFORM STREQUAL "nvcc")
    target_link_libraries(hiprand PRIVATE "-lcurand")
else()
    add_dependencies(hiprand rocrand)
    target_link_libraries(hiprand
        PRIVATE
            -Wl,-rpath,${rocRAND_rpath} # workaround for HIP linker
            $<TARGET_FILE:rocrand>
    )
    foreach(amdgpu_target ${AMDGPU_TARGETS})
        target_link_libraries(hiprand PRIVATE --amdgpu-target=${amdgpu_target})
    endforeach()
endif()
set_target_properties(hiprand
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
        DEBUG_POSTFIX "-d"
        VERSION ${hiprand_VERSION}
        SOVERSION 1
)

# Install
# .so library
install(
    TARGETS hiprand
    EXPORT hiprand-targets
    LIBRARY DESTINATION hiprand/lib
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)
# headers
install(
    DIRECTORY
        "include/"
        "${PROJECT_BINARY_DIR}/library/include/"
    DESTINATION hiprand/include
    FILES_MATCHING
    PATTERN "hiprand*h"
    PATTERN "hiprand*hpp"
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
# .cmake files
install(
    EXPORT hiprand-targets
    NAMESPACE hip::
    DESTINATION hiprand/lib/cmake/hiprand
)
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/hiprand/include")
set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/hiprand/lib")
include(CMakePackageConfigHelpers)
configure_package_config_file(
    src/hiprand-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/hiprand
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config-version.cmake
    VERSION ${hiprand_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config-version.cmake
    DESTINATION hiprand/lib/cmake/hiprand
)

if (BUILD_FORTRAN_WRAPPER)
    add_subdirectory(src/fortran)
endif()

# Package (make package)
# Generates .deb, .zip., and .tar.gz packages
find_program(RPMBUILD_EXE rpmbuild)
find_program(DPKG_EXE dpkg)
set(CPACK_GENERATOR "TGZ;ZIP")
if(EXISTS ${RPMBUILD_EXE})
    list(APPEND CPACK_GENERATOR "RPM")
endif()
if(EXISTS ${DPKG_EXE})
    list(APPEND CPACK_GENERATOR "DEB")
endif()
set(CPACK_PACKAGE_NAME "rocrand")
set(CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Jakub Szuppe <jakub@streamhpc.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The rocRAND library provides functions that generate pseudo-random and quasi-random numbers.")
set(CPACK_SET_DESTDIR ON)
set(CPACK_PACKAGE_VERSION_MAJOR ${rocrand_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${rocrand_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${rocrand_VERSION_PATCH})
set(LIB_NAMES "hiprand" "rocrand")
set(LIB_DIRS "${CMAKE_INSTALL_PREFIX}/hiprand/lib" "${CMAKE_INSTALL_PREFIX}/rocrand/lib")
package_set_postinst_prerm(
    "hiprand;rocrand"
    "${CMAKE_INSTALL_PREFIX}/hiprand/lib;${CMAKE_INSTALL_PREFIX}/rocrand/lib"
    "${CMAKE_INSTALL_PREFIX}/hiprand/include;${CMAKE_INSTALL_PREFIX}/rocrand/include"
)

# Include CPack to introduce the appropriate targets
include(CPack)
