# rocRAND library
# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(rocRAND_SRCS ${tmp})

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
    "${PROJECT_SOURCE_DIR}/library/include/rocrand/rocrand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/rocrand/rocrand_version.h"
    @ONLY
)

add_library(rocrand ${rocRAND_SRCS})
add_library(roc::rocrand ALIAS rocrand)

target_include_directories(rocrand INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/rocrand>)

# Build library
if(HIP_COMPILER STREQUAL "nvcc")
    set_source_files_properties(${rocRAND_SRCS}
        PROPERTIES LANGUAGE CUDA
    )
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
    target_link_libraries(rocrand PRIVATE hip::device)
    if(NOT WIN32)
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(rocrand PRIVATE --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
endif()

rocm_set_soversion(rocrand ${rocrand_SOVERSION})
set_target_properties(rocrand
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
        DEBUG_POSTFIX "-d"
)

rocm_install(
    TARGETS rocrand
    INCLUDE
        "${CMAKE_SOURCE_DIR}/library/include"
        "${CMAKE_BINARY_DIR}/library/include"
)

set(FORTRAN_SRCS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/rocrand/src/fortran")
configure_file(
    src/rocrand-fortran-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake
)


if(HIP_COMPILER STREQUAL "nvcc")
    rocm_export_targets(
        TARGETS roc::rocrand
        NAMESPACE roc::
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake"
    )
else()
    rocm_export_targets(
        TARGETS roc::rocrand
        NAMESPACE roc::
        DEPENDS PACKAGE hip
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake"
    )
endif()

# install library to C:\hipSDK\bin
if (WIN32)
    install (TARGETS rocrand DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
    if (BUILD_TEST)
        install (TARGETS rocrand DESTINATION "${PROJECT_BINARY_DIR}/test")
    endif()
endif()

# Fortran wrappers for hipRAND and rocRAND
if(BUILD_FORTRAN_WRAPPER)
    add_subdirectory(src/fortran)
endif()

# Package (make package)
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.txt")
if(WIN32)
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_GENERATOR "ZIP")
    set(CMAKE_INSTALL_PREFIX "C:/hipSDK" CACHE PATH "Install path" FORCE)
    set(INSTALL_PREFIX "C:/hipSDK")
    set(CPACK_SET_DESTDIR OFF)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY "C:/hipSDK")
    set(CPACK_PACKAGING_INSTALL_PREFIX "")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY OFF)
else()
    if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
        set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
    endif()
endif()

if( HIP_RUNTIME_LOWER STREQUAL "rocclr" )
    rocm_package_add_dependencies("hip-rocclr >= 3.5.0")
elseif( HIP_RUNTIME STREQUAL "cuda" )
    rocm_package_add_dependencies("hip-nvcc >= 3.5.0")
endif( )

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

if(HIP_COMPILER STREQUAL "nvcc")
    set(package_name rocrand-alt)
else()
    set(package_name rocrand)
endif()

set(ROCRAND_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

if(WIN32)
    option(BUILD_HIPRAND "Build hipRAND as part of rocRAND" OFF)
else()
    option(BUILD_HIPRAND "Build hipRAND as part of rocRAND" ON)
endif()
if(BUILD_HIPRAND)
    if(EXISTS ${CMAKE_SOURCE_DIR}/hipRAND/CMakeLists.txt)
        set(HIPRAND_SUBMODULE ON CACHE BOOL "Build hipRAND as a submodule of rocRAND." FORCE)
        add_subdirectory(${CMAKE_SOURCE_DIR}/hipRAND ${CMAKE_BINARY_DIR}/hipRAND)
    else()
        message(FATAL_ERROR "BUILD_HIPRAND was set, but the hipRAND submodule could not be found. Use git submodule update --init to clone the hipRAND submodule, or set BUILD_HIPRAND to OFF (--no-hiprand flag for ./install)")
    endif()
endif()

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "Radeon Open Compute RAND library"
    MAINTAINER "rocRAND Maintainer <hiprand-maintainer@amd.com>"
    LDCONFIG
    # LDCONFIG_DIR ${ROCRAND_CONFIG_DIR}
)
