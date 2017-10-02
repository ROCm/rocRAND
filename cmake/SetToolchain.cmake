# Find HIP package
find_package(HIP REQUIRED)

# Select toolchain
if(HIP_PLATFORM STREQUAL "nvcc")
    set(CMAKE_C_COMPILER gcc)
    set(CMAKE_CXX_COMPILER g++)
elseif(HIP_PLATFORM STREQUAL "hcc")
    # Find HCC executable
    find_program(
        HIP_HCC_EXECUTABLE
        NAMES hcc
        PATHS
        "${HIP_ROOT_DIR}"
        ENV ROCM_PATH
        ENV HIP_PATH
        /opt/rocm
        /opt/rocm/hip
        PATH_SUFFIXES bin
        NO_DEFAULT_PATH
        )
    if(NOT HIP_HCC_EXECUTABLE)
        # Now search in default paths
        find_program(HIP_HCC_EXECUTABLE hipcc)
    endif()
    mark_as_advanced(HIP_HCC_EXECUTABLE)
    set(CMAKE_CXX_COMPILER ${HIP_HCC_EXECUTABLE})
else()
    message(FATAL_ERROR "HIP_PLATFORM must be 'nvcc' (CUDA) or 'hcc' (ROCm).")
endif()
