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

include(CMakePackageConfigHelpers)

add_library(rocrand ${rocRAND_SRCS})

# Build library
if(HIP_COMPILER STREQUAL "nvcc")
    set_source_files_properties(${rocRAND_SRCS}
        PROPERTIES LANGUAGE CUDA
    )
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
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
        target_link_libraries(rocrand PRIVATE --amdgpu-target=${amdgpu_target})
    endforeach()
endif()

set(rocrandtargets rocrand)

# Install
# .so or .a lib
rocm_install(
    TARGETS ${rocrandtargets}
    EXPORT rocrand-targets
    ARCHIVE DESTINATION rocrand/lib
    LIBRARY DESTINATION rocrand/lib
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)
# headers
rocm_install(
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
rocm_install(
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
rocm_install(
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
    set_source_files_properties(${hipRAND_SRCS} PROPERTIES LANGUAGE CUDA)
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
endif()
add_library(hiprand ${hipRAND_SRCS})
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
        target_link_libraries(hiprand PRIVATE --amdgpu-target=${amdgpu_target})
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
rocm_install(
    TARGETS ${hiprandtargets}
    EXPORT hiprand-targets
    ARCHIVE DESTINATION hiprand/lib
    LIBRARY DESTINATION hiprand/lib
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)
# headers
rocm_install(
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
rocm_install(
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
rocm_install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config-version.cmake
    DESTINATION hiprand/lib/cmake/hiprand
)

# Fortran wrappers for hipRAND and rocRAND
if(BUILD_FORTRAN_WRAPPER)
    add_subdirectory(src/fortran)
endif()

# Package specific CPACK vars
# As of ROCm 3.8, HIP_RUNTIME has been changed from "ROCclr" to "rocclr"
# As of ROCm 3.8, HIP_RUNTIME has been changed from "cuda" to "cudart"
string( TOLOWER "${HIP_RUNTIME}" HIP_RUNTIME_LOWER )
if( HIP_RUNTIME_LOWER STREQUAL "rocclr" )
  set( RPM_REQUIREMENTS "hip-rocclr >= 3.5.0" )
  set( CPACK_DEBIAN_PACKAGE_DEPENDS "hip-rocclr (>= 3.5.0)" )
elseif( HIP_RUNTIME MATCHES ".*cuda.*")
  set( RPM_REQUIREMENTS "hip-nvcc >= 3.5.0" )
  set( CPACK_DEBIAN_PACKAGE_DEPENDS "hip-nvcc (>= 3.5.0)" )
else()
  set( RPM_REQUIREMENTS "rocm-dev >= 2.5.27" )
  set( CPACK_DEBIAN_PACKAGE_DEPENDS "rocm-dev (>= 2.5.27)" )
endif()

set( CPACK_RPM_PACKAGE_REQUIRES ${RPM_REQUIREMENTS} )

set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.txt" )

if (WIN32)
  SET( CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE )
  SET( INSTALL_PREFIX "C:/hipSDK" )
  SET( CPACK_SET_DESTDIR FALSE )
  SET( CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK" )
  SET( CPACK_PACKAGING_INSTALL_PREFIX "" )
  set( CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF )
endif( )

set( CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include" "\${CPACK_PACKAGING_INSTALL_PREFIX}/lib" )

# Give rocblas compiled for CUDA backend a different name
if( CMAKE_CXX_COMPILER_ID MATCHES "Clang" )
    set( package_name rocrand )
else( )
    set( package_name rocrand-alt )
endif( )

set( ROCRAND_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file.")

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "the rocRAND project provides functions that generate pseudo-random and quasi-random numbers."
    MAINTAINER "rocRAND Maintainer <rocrand-maintainer@amd.com>"
    LDCONFIG
    LDCONFIG_DIR ${ROCRAND_CONFIG_DIR}
)