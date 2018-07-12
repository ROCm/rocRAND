# Benchmarks

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rocrand*.cpp)
set(rocRAND_BENCHMARK_SRCS ${tmp})
if(HIP_PLATFORM STREQUAL "nvcc")
    file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_curand*.cpp)
    set(rocRAND_BENCHMARK_SRCS ${rocRAND_BENCHMARK_SRCS} ${tmp})
endif()

# Use CUDA_INCLUDE_DIRECTORIES to include required dirs
# for nvcc if cmake version is less than 3.9.3# Use CUDA_INCLUDE_DIRS if cmake version is less than 3.9.3
if((HIP_PLATFORM STREQUAL "nvcc") AND (CMAKE_VERSION VERSION_LESS "3.9.3"))
    CUDA_INCLUDE_DIRECTORIES(
        "${PROJECT_BINARY_DIR}/library/include/"
        "${PROJECT_SOURCE_DIR}/library/include/"
    )
endif()

# Build benchmarks
foreach(benchmark_src ${rocRAND_BENCHMARK_SRCS})
    get_filename_component(benchmark_name ${benchmark_src} NAME_WE)
    # nvcc/CUDA
    if(HIP_PLATFORM STREQUAL "nvcc")
        set_source_files_properties(${benchmark_src}
            PROPERTIES
                CUDA_SOURCE_PROPERTY_FORMAT OBJ
        )
        CUDA_ADD_EXECUTABLE(${benchmark_name}
            ${benchmark_src}
            OPTIONS --expt-extended-lambda
        )
    # hcc/ROCm
    else()
        add_executable(${benchmark_name} ${benchmark_src})
    endif()
    if(HIP_PLATFORM STREQUAL "nvcc")
        target_link_libraries(${benchmark_name}
            rocrand
            ${CUDA_curand_LIBRARY}
        )
    else()
        # Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
        if(TARGET hip::device)
            target_link_libraries(${benchmark_name}
                rocrand
                hip::device
            )
        else()
            target_link_libraries(${benchmark_name}
                rocrand
                hip::hip_hcc hip::hip_device
            )
        endif()
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${benchmark_name} --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(${benchmark_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
    )
endforeach()
