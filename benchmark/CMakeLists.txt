# Benchmarks

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rocrand*.cpp)
set(rocRAND_BENCHMARK_SRCS ${tmp})
if(HIP_PLATFORM STREQUAL "nvcc")
    file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_curand*.cpp)
    set(rocRAND_BENCHMARK_SRCS ${rocRAND_BENCHMARK_SRCS} ${tmp})
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
            ${TestU01_LIBRARIES}
        )
    else()
        target_link_libraries(${benchmark_name}
            rocrand
            hip::hip_hcc hip::hip_device
            ${TestU01_LIBRARIES}
        )
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${benchmark_name} --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(${benchmark_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
    )
endforeach()
