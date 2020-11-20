# rocRAND library

set( lib_SOVERSION 1.1 )
set( hiprand_SOVERSION ${lib_SOVERSION} )
set( rocrand_SOVERSION ${lib_SOVERSION} )

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(rocRAND_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/*.cpp)
foreach(to_exclude ${tmp})
    list(REMOVE_ITEM rocRAND_SRCS "${to_exclude}")
endforeach()

# When enabled, it defines ROCRAND_ENABLE_INLINE_ASM in rocrand_version.h, which
# turns on inline asm in rocRAND (for both compiled library and device functions).
option(ENABLE_INLINE_ASM "Enable inline asm optimisations in rocRAND" ON)
if(ENABLE_INLINE_ASM)
    set(
        rocrand_ENABLE_INLINE_ASM
        "\n// Enables inline asm optimisations\n"
        "#if !defined(ROCRAND_ENABLE_INLINE_ASM) && !defined(ROCRAND_DISABLE_INLINE_ASM)\n"
        "    #define ROCRAND_ENABLE_INLINE_ASM\n"
        "#endif"
    )
    string(REPLACE ";" "" rocrand_ENABLE_INLINE_ASM "${rocrand_ENABLE_INLINE_ASM}")
endif()

# Configure a header file to pass the rocRAND version
configure_file(
    "${PROJECT_SOURCE_DIR}/library/include/rocrand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/rocrand_version.h"
    @ONLY
)

# Use CUDA_INCLUDE_DIRECTORIES to include required dirs
# for nvcc if cmake version is less than 3.9.3
if((HIP_COMPILER STREQUAL "nvcc") AND (CMAKE_VERSION VERSION_LESS "3.9.3"))
    CUDA_INCLUDE_DIRECTORIES(
        "${PROJECT_BINARY_DIR}/library/include/"
        "${PROJECT_SOURCE_DIR}/library/include/"
    )
endif()

include(CMakePackageConfigHelpers)

# Build library
if(HIP_COMPILER STREQUAL "nvcc")
    set_source_files_properties(${rocRAND_SRCS}
        PROPERTIES
            CUDA_SOURCE_PROPERTY_FORMAT OBJ
    )
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    CUDA_ADD_LIBRARY(rocrand ${rocRAND_SRCS})
else()
    add_library(rocrand ${rocRAND_SRCS})

    # Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
    if(CXX_VERSION_STRING MATCHES "clang")
        target_link_libraries(rocrand PRIVATE hip::device)
    else()
      target_link_libraries(rocrand
          PRIVATE
              # We keep hip::device private, because otherwise it's not possible
              # to link to roc::rocrand when using different compiler than hcc,
              # hip::device adds hcc-specific compilation flags.
              hip::device
      )
    endif()
    set(rocrand_DEPENDENCIES "hip")
endif()

target_include_directories(rocrand
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
        $<INSTALL_INTERFACE:rocrand/include>
)

rocm_set_soversion(rocrand ${rocrand_SOVERSION})
set_target_properties(rocrand
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
        DEBUG_POSTFIX "-d"
)

# Set AMD targets
if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
    foreach(amdgpu_target ${AMDGPU_TARGETS})
        target_link_libraries(rocrand PRIVATE --cuda-gpu-arch=${amdgpu_target})
    endforeach()
endif()

set(rocrandtargets rocrand)

# Install
# .so or .a lib
install(
    TARGETS ${rocrandtargets}
    EXPORT rocrand-targets
    ARCHIVE DESTINATION rocrand/lib
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
set(FORTRAN_SRCS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/rocrand/src/fortran")
configure_package_config_file(
    src/rocrand-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/rocrand-config.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/rocrand
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR FORTRAN_SRCS_INSTALL_DIR
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

# Get hipRAND sources
if(HIP_COMPILER STREQUAL "nvcc")
    set(hipRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/hiprand_${HIP_COMPILER}.cpp)
else()
    set(hipRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/hiprand.cpp)
endif()

# Configure a header file to pass the hipRAND version
configure_file(
    "${PROJECT_SOURCE_DIR}/library/include/hiprand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/hiprand_version.h"
    @ONLY
)

# Build
if(HIP_COMPILER STREQUAL "nvcc")
    set_source_files_properties(${hipRAND_SRCS}
        PROPERTIES
            CUDA_SOURCE_PROPERTY_FORMAT OBJ
    )
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    CUDA_ADD_LIBRARY(hiprand ${hipRAND_SRCS})
else()
    add_library(hiprand ${hipRAND_SRCS})
endif()
target_include_directories(hiprand
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/library/include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/include>
        $<INSTALL_INTERFACE:hiprand/include>
)

if(HIP_COMPILER STREQUAL "nvcc")
    target_link_libraries(hiprand
        ${CUDA_curand_LIBRARY}
    )
else()
    # Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
    if(CXX_VERSION_STRING MATCHES "clang")
        target_link_libraries(hiprand PRIVATE rocrand hip::device)
    else()
      target_link_libraries(hiprand PRIVATE rocrand hip::device)
    endif()
    foreach(amdgpu_target ${AMDGPU_TARGETS})
        target_link_libraries(hiprand PRIVATE --cuda-gpu-arch=${amdgpu_target})
    endforeach()
endif()

rocm_set_soversion(hiprand ${hiprand_SOVERSION})
set_target_properties(hiprand
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
        DEBUG_POSTFIX "-d"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/rocrand/lib"
)

set(hiprandtargets hiprand)

# Install
# .so and .a library
install(
    TARGETS ${hiprandtargets}
    EXPORT hiprand-targets
    ARCHIVE DESTINATION hiprand/lib
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
set(FORTRAN_SRCS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/hiprand/src/fortran")
configure_package_config_file(
    src/hiprand-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/cmake/hiprand
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR FORTRAN_SRCS_INSTALL_DIR
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

# Fortran wrappers for hipRAND and rocRAND
if(BUILD_FORTRAN_WRAPPER)
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
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "rocRAND Maintainer <rocrand-maintaner@amd.com>")
if(HIP_RUNTIME STREQUAL "ROCclr")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-rocclr (>= 3.5.0)")
  set(CPACK_RPM_PACKAGE_REQUIRES "hip-rocclr >= 3.5.0")
elseif(HIP_RUNTIME STREQUAL "cuda")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip-nvcc (>= 3.5.0)")
  set(CPACK_RPM_PACKAGE_REQUIRES "hip-nvcc >= 3.5.0")
endif()
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The rocRAND library provides functions that generate pseudo-random and quasi-random numbers.")
set(CPACK_SET_DESTDIR ON)
set(CPACK_RPM_PACKAGE_RELOCATABLE OFF)
set(CPACK_RPM_PACKAGE_AUTOREQPROV OFF CACHE BOOL "")
set(CPACK_PACKAGE_VERSION ${rocRAND_VERSION})
set(CPACK_PACKAGE_VERSION_MAJOR ${rocRAND_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${rocRAND_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${rocRAND_VERSION_PATCH})

# This section copied from rocm-cmake project per terms of the MIT license
# rocRAND packaging will switch to use rocm-cmake macros in the future
if (EXISTS "/etc/os-release")
  rocm_set_os_id(_os_id)
  rocm_read_os_release(_version_id "VERSION_ID")

  #only set CPACK_SYSTEM_NAME for AMD supported OSes
  if (_os_id_centos OR _os_is_rhel)
    STRING(CONCAT _SYSTEM_NAME "el" ${_version_id} ".x86_64")
    #Debs use underscrore between OS and architecture
  elseif(_os_id_ubuntu)
    STRING(CONCAT _SYSTEM_NAME ${_os_id} "-" ${_version_id} "_amd64")
  else()
    #For SLES and unsupported OSes
    STRING(CONCAT _SYSTEM_NAME ${_os_id} "-" ${_version_id} ".amd64")
  endif()

  set(CPACK_SYSTEM_NAME ${_SYSTEM_NAME} CACHE STRING "CPACK_SYSTEM_NAME for packaging")
endif()

package_set_postinst_prerm(
    "hiprand;rocrand"
    "${CMAKE_INSTALL_PREFIX}/hiprand/lib;${CMAKE_INSTALL_PREFIX}/rocrand/lib"
    "${CMAKE_INSTALL_PREFIX}/hiprand/include;${CMAKE_INSTALL_PREFIX}/rocrand/include"
    "${hiprand_SOVERSION};${rocrand_SOVERSION}"
)

# Include CPack to introduce the appropriate targets
include(CPack)
