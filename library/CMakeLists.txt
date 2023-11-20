# MIT License
#
# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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

set_target_properties(rocrand PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
)

# Add interface include directory so that other CMake applications can maintain previous behaviour.
# This will be removed with upcoming packaging changes.
target_include_directories(rocrand INTERFACE $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include/rocrand>)

# Build library
if(NOT USE_HIP_CPU)
    if(HIP_COMPILER STREQUAL "nvcc")
        set_source_files_properties(${rocRAND_SRCS}
            PROPERTIES LANGUAGE CUDA
        )
        set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    else()
        target_link_libraries(rocrand PRIVATE hip::device)
    endif()
else()
    # Check hipRuntimeGetVersion() function availability
    file(WRITE "${PROJECT_BINARY_DIR}/hip_cpu_test/test_runtime_get_version.cpp"
        [[
            #include <hip/hip_runtime.h>
            int main()
            {
                int runtime_version;
                hipRuntimeGetVersion(&runtime_version);
            }
        ]]
    )
    # Check HIP min() function availability
    file(WRITE "${PROJECT_BINARY_DIR}/hip_cpu_test/test_hip_math.cpp"
        [[
            #include <hip/hip_runtime.h>
            int main()
            {
                return min(2, 3);
            }
        ]]
    )
    try_compile(HAVE_HIP_RUNTIME_GET_VERSION
        "${PROJECT_BINARY_DIR}/hip_cpu_test"
        "${PROJECT_BINARY_DIR}/hip_cpu_test/test_runtime_get_version.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_BINARY_DIR}/deps/hip-cpu/include"
        CXX_STANDARD 17
        LINK_LIBRARIES Threads::Threads
    )
    try_compile(HAVE_HIP_MATH
        "${PROJECT_BINARY_DIR}/hip_cpu_test"
        "${PROJECT_BINARY_DIR}/hip_cpu_test/test_hip_math.cpp"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_BINARY_DIR}/deps/hip-cpu/include"
        CXX_STANDARD 17
        LINK_LIBRARIES Threads::Threads
    )
    target_link_libraries(rocrand
        Threads::Threads
        hip_cpu_rt::hip_cpu_rt
    )
    if(STL_DEPENDS_ON_TBB)
        target_link_libraries(rocrand TBB::tbb)
    endif()
endif()

# Configure a header file for min() and hipRuntimeGetVersion() functions
configure_file(
    "${PROJECT_SOURCE_DIR}/library/include/rocrand/rocrand_hip_cpu.h.in"
    "${PROJECT_BINARY_DIR}/library/include/rocrand/rocrand_hip_cpu.h"
    @ONLY
)

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
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR})
set(CONFIG_PACKAGE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/rocrand)

include(CMakePackageConfigHelpers)
configure_package_config_file(
    src/rocrand-fortran-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake
    INSTALL_DESTINATION ${CONFIG_PACKAGE_INSTALL_DIR}
)

if(USE_HIP_CPU)
    rocm_export_targets(
        TARGETS roc::rocrand
        NAMESPACE roc::
        DEPENDS PACKAGE Threads
        DEPENDS PACKAGE hip_cpu_rt
        DEPENDS PACKAGE TBB
        INCLUDE "${CMAKE_CURRENT_BINARY_DIR}/rocrand-fortran-config.cmake"
    )
elseif(HIP_COMPILER STREQUAL "nvcc")
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

# Fortran wrappers for rocRAND
if(BUILD_FORTRAN_WRAPPER)
    add_subdirectory(src/fortran)
endif()
