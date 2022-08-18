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

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
    rocm_wrap_header_file(
        rocrand_version.h
        GUARDS SYMLINK WRAPPER
        WRAPPER_LOCATIONS include rocrand/include
        OUTPUT_LOCATIONS library/include library/rocrand/include
        HEADER_LOCATION include/rocrand
    )
endif()

add_library(rocrand ${rocRAND_SRCS})
add_library(roc::rocrand ALIAS rocrand)

# Add interface include directory so that other CMake applications can maintain previous behaviour.
# This will be removed with upcoming packaging changes.
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

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY AND NOT WIN32)
    rocm_install(
        DIRECTORY "${PROJECT_BINARY_DIR}/library/rocrand"
        DESTINATION "."
    )
endif()

# install library to C:\hipSDK\bin
if (WIN32)
    install (TARGETS rocrand DESTINATION "${CMAKE_INSTALL_PREFIX}/bin")
    if (BUILD_TEST)
	    add_custom_command(
		    TARGET rocrand 
		    POST_BUILD
		    COMMAND ${CMAKE_COMMAND} -E copy
			    $<TARGET_FILE:rocrand>
			    ${PROJECT_BINARY_DIR}/test/$<TARGET_FILE_NAME:rocrand>
	    )
    endif()
    if (BUILD_BENCHMARK)
	    add_custom_command(
		    TARGET rocrand 
		    POST_BUILD
		    COMMAND ${CMAKE_COMMAND} -E copy
			    $<TARGET_FILE:rocrand>
			    ${PROJECT_BINARY_DIR}/benchmark/$<TARGET_FILE_NAME:rocrand>
	    )
    endif()	
endif()

# Fortran wrappers for hipRAND and rocRAND
if(BUILD_FORTRAN_WRAPPER)
    add_subdirectory(src/fortran)
endif()
