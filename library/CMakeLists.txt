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
    "${PROJECT_SOURCE_DIR}/library/include/rocrand/rocrand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/rocrand/rocrand_version.h"
    @ONLY
)

add_library(rocrand ${rocRAND_SRCS})
add_library(roc::rocrand ALIAS rocrand)

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
    PREFIX rocrand
)

set(FORTRAN_SRCS_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/rocrand/src/fortran")
configure_file(
    src/rocrand-fortran-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake
)
configure_file(
    src/hiprand-fortran-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/hiprand-fortran-config.cmake
)

# hipRAND library
# Get hipRAND sources
if(HIP_COMPILER STREQUAL "nvcc")
    set(hipRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/hiprand_nvcc.cpp)
    set_source_files_properties(${hipRAND_SRCS} PROPERTIES LANGUAGE CUDA)
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
else()
    set(hipRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/hiprand.cpp)
endif()

# Configure a header file to pass the hipRAND version
configure_file(
    "${PROJECT_SOURCE_DIR}/library/include/hiprand/hiprand_version.h.in"
    "${PROJECT_BINARY_DIR}/library/include/hiprand/hiprand_version.h"
    @ONLY
)

# Build
add_library(hiprand ${hipRAND_SRCS})

if(HIP_COMPILER STREQUAL "nvcc")
    target_link_libraries(hiprand
        ${CUDA_curand_LIBRARY}
    )
else()
    target_link_libraries(hiprand PRIVATE roc::rocrand hip::device)
	if (NOT WIN32)
      foreach(amdgpu_target ${AMDGPU_TARGETS})
          target_link_libraries(hiprand PRIVATE --amdgpu-target=${amdgpu_target})
      endforeach()
	endif()
endif()

rocm_set_soversion(hiprand ${hiprand_SOVERSION})
set_target_properties(hiprand
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
        DEBUG_POSTFIX "-d"
        INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/rocrand/lib"
)

rocm_install(
    TARGETS hiprand
    INCLUDE
        "${CMAKE_SOURCE_DIR}/library/include"
        "${CMAKE_BINARY_DIR}/library/include"
    PREFIX hiprand
    EXPORT hiprand-targets
)

if(HIP_COMPILER STREQUAL "nvcc")
    rocm_export_targets(
        TARGETS roc::rocrand
        PREFIX rocrand
        NAMESPACE roc::
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake"
    )
    rocm_export_targets(
        TARGETS hip::hiprand
        NAME hiprand
        PREFIX hiprand
        NAMESPACE hip::
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/hiprand-fortran-config.cmake"
    )
else()
    rocm_export_targets(
        TARGETS roc::rocrand
        PREFIX rocrand
        NAMESPACE roc::
        DEPENDS PACKAGE hip
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake"
    )
    rocm_export_targets(
        TARGETS hip::hiprand
        NAME hiprand
        PREFIX hiprand
        NAMESPACE hip::
        DEPENDS PACKAGE hip
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/hiprand-fortran-config.cmake"
    )
endif()

# install library to C:\hipSDK\bin
if (WIN32)
    install (TARGETS rocrand hiprand DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
    if (BUILD_TEST)
        install (TARGETS rocrand hiprand DESTINATION "${PROJECT_BINARY_DIR}/test")
    endif()
endif()

# Fortran wrappers for hipRAND and rocRAND
if(BUILD_FORTRAN_WRAPPER)
    add_subdirectory(src/fortran)
endif()

function(install_symlinks SRC_DIR DEST_DIR)
    # TODO: Check if SUBDIR is relative path Copy instead of symlink on windows
    if(CMAKE_HOST_WIN32)
        set(SYMLINK_CMD "file(COPY \${SRC} DESTINATION \${DEST_DIR})")
    else()
        set(SYMLINK_CMD "execute_process(COMMAND ln -sf \${SRC_REL} \${DEST})")
    endif()

    set(INSTALL_CMD "
        set(SRC_DIR \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${SRC_DIR})
        set(DEST_DIR \$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/${DEST_DIR})
        file(GLOB_RECURSE FILES FOLLOW_SYMLINKS RELATIVE \${SRC_DIR} \${SRC_DIR}/*)
        foreach(FILE \${FILES})
            set(SRC \${SRC_DIR}/\${FILE})
            set(DEST \${DEST_DIR}/\${FILE})
            get_filename_component(DEST_PARENT \${DEST} DIRECTORY)
            file(MAKE_DIRECTORY \${DEST_PARENT})
            file(RELATIVE_PATH SRC_REL \${DEST_PARENT} \${SRC})
            message(STATUS \"symlink: \${SRC_REL} -> \${DEST}\")
            ${SYMLINK_CMD}
        endforeach()
    ")
    rocm_install(CODE "${INSTALL_CMD}")
endfunction()
install_symlinks(rocrand/include/rocrand rocrand/include)
install_symlinks(hiprand/include/hiprand hiprand/include)

rocm_install_symlink_subdir(rocrand .)
rocm_install_symlink_subdir(hiprand .)


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

    rocm_join_if_set(", " CPACK_DEBIAN_RUNTIME_PACKAGE_PROVIDES "hiprand")
    rocm_join_if_set(", " CPACK_RPM_RUNTIME_PACKAGE_PROVIDES "hiprand")
    rocm_join_if_set(", " CPACK_DEBIAN_DEVEL_PACKAGE_PROVIDES "hiprand-dev")
    rocm_join_if_set(", " CPACK_RPM_DEVEL_PACKAGE_PROVIDES "hiprand-devel")
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

rocm_create_package(
    NAME ${package_name}
    DESCRIPTION "Radeon Open Compute RAND library"
    MAINTAINER "rocRAND Maintainer <hiprand-maintainer@amd.com>"
    LDCONFIG
    # LDCONFIG_DIR ${ROCRAND_CONFIG_DIR}
)
