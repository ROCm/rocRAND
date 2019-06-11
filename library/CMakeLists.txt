# rocRAND library

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
    "${PROJECT_SOURCE_DIR}/library/include/rocrand/rocrand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/rocrand/rocrand_version.h"
    @ONLY
)

# Use CUDA_INCLUDE_DIRECTORIES to include required dirs
# for nvcc if cmake version is less than 3.9.3
if((HIP_PLATFORM STREQUAL "nvcc") AND (CMAKE_VERSION VERSION_LESS "3.9.3"))
    CUDA_INCLUDE_DIRECTORIES(
        "${PROJECT_BINARY_DIR}/library/include/"
        "${PROJECT_SOURCE_DIR}/library/include/"
    )
endif()

# Build library
if(HIP_PLATFORM STREQUAL "nvcc")
    set_source_files_properties(${rocRAND_SRCS}
        PROPERTIES
            CUDA_SOURCE_PROPERTY_FORMAT OBJ
    )
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    CUDA_ADD_LIBRARY(rocrand ${rocRAND_SRCS})
else()
    add_library(rocrand ${rocRAND_SRCS})
    target_link_libraries(rocrand PRIVATE hip::device)
endif()

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
# Set AMD targets
if(HIP_PLATFORM STREQUAL "hcc")
    foreach(amdgpu_target ${AMDGPU_TARGETS})
        target_link_libraries(rocrand PRIVATE --amdgpu-target=${amdgpu_target})
    endforeach()
endif()

# Install
# .so lib
rocm_install_targets(
    TARGETS rocrand
    EXPORT rocrand-targets
    PREFIX rocrand
)
# headers
install(
    DIRECTORY
        "include/rocrand"
        "${PROJECT_BINARY_DIR}/library/include/rocrand"
    DESTINATION rocrand/include
    FILES_MATCHING
    PATTERN "rocrand*h"
    PATTERN "rocrand*hpp"
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
# .cmake files
rocm_export_targets(
    NAME rocrand
    TARGETS roc::rocrand
    PREFIX rocrand
    DEPENDS PACKAGE HIP
    NAMESPACE roc::
)
set(FORTRAN_SRCS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/rocrand/src/fortran")
# Append Fortran wrapper path to the config file  genereated by rocm_export_targets
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/rocrand-config.cmake "
# Fortran wrapper
if(${BUILD_FORTRAN_WRAPPER})
    set_and_check(rocrand_FORTRAN_SRC_DIR \"${FORTRAN_SRCS_INSTALL_DIR}\")
    set_and_check(rocrand_FORTRAN_SRC_DIRS \"${FORTRAN_SRCS_INSTALL_DIR}\")
    set(rocrand_FORTRAN_FOUND YES)
else()
    set(rocrand_FORTRAN_FOUND NOTFOUND)
endif()
")

# Create symlinks
rocm_install_symlink_subdir(rocrand)


# hipRAND library

# Get hipRAND sources
set(hipRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/hiprand_${HIP_PLATFORM}.cpp)

# Configure a header file to pass the hipRAND version
configure_file(
    "${PROJECT_SOURCE_DIR}/library/include/hiprand/hiprand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/hiprand/hiprand_version.h"
    @ONLY
)

# Build
if(HIP_PLATFORM STREQUAL "nvcc")
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
if(HIP_PLATFORM STREQUAL "nvcc")
    target_link_libraries(hiprand
        ${CUDA_curand_LIBRARY}
    )
else()
    target_link_libraries(hiprand PRIVATE rocrand hip::device)
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
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/rocrand/lib"
)

# Install
# .so lib
rocm_install_targets(
    TARGETS hiprand
    EXPORT hiprand-targets
    PREFIX hiprand
)
# headers
install(
    DIRECTORY
        "include/hiprand"
        "${PROJECT_BINARY_DIR}/library/include/hiprand"
    DESTINATION hiprand/include
    FILES_MATCHING
    PATTERN "hiprand*h"
    PATTERN "hiprand*hpp"
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
# .cmake files
rocm_export_targets(
    NAME hiprand
    TARGETS hip::hiprand
    PREFIX hiprand
    DEPENDS PACKAGE HIP
    NAMESPACE hip::
)
set(FORTRAN_SRCS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/hiprand/src/fortran")
# Append Fortran wrapper path to the config file  genereated by rocm_export_targets
file(APPEND ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config.cmake "
# Fortran wrapper
if(${BUILD_FORTRAN_WRAPPER})
    set_and_check(hiprand_FORTRAN_SRC_DIR \"${FORTRAN_SRCS_INSTALL_DIR}\")
    set_and_check(hiprand_FORTRAN_SRC_DIRS \"${FORTRAN_SRCS_INSTALL_DIR}\")
    set(hiprand_FORTRAN_FOUND YES)
else()
    set(hiprand_FORTRAN_FOUND NOTFOUND)
endif()
")

# Create symlinks
rocm_install_symlink_subdir(hiprand)


# Fortran wrappers for hipRAND and rocRAND
if(BUILD_FORTRAN_WRAPPER)
    add_subdirectory(src/fortran)
endif()


# Package (make package)
# Generates .deb, .zip., and .tar.gz packages
if(HIP_PLATFORM STREQUAL "hcc")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.5.19055)") # 1.5.19055 is part of ROCm 2.2
    set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.5.19055") # 1.5.19055 is part of ROCm 2.2
    rocm_create_package(
        NAME rocrand
        DESCRIPTION "The rocRAND library provides functions that generate pseudo-random and quasi-random numbers."
        MAINTAINER "Saad Rahim <saad.rahim@amd.com>"
    )
else()
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_nvcc (>= 1.5.19055)") # 1.5.19055 is part of ROCm 2.2
    set(CPACK_RPM_PACKAGE_REQUIRES "hip_nvcc >= 1.5.19055") # 1.5.19055 is part of ROCm 2.2
    rocm_create_package(
        NAME rocrand
        DESCRIPTION "The rocRAND library provides functions that generate pseudo-random and quasi-random numbers."
        MAINTAINER "Saad Rahim <saad.rahim@amd.com>"
    )
endif()
