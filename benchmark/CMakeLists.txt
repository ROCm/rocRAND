# Benchmarks

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rocrand*.cpp)
set(rocRAND_BENCHMARK_SRCS ${tmp})
if(HIP_PLATFORM STREQUAL "nvcc")
    file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_curand*.cpp)
    set(rocRAND_BENCHMARK_SRCS ${rocRAND_BENCHMARK_SRCS} ${tmp})
endif()
set_source_files_properties(${rocRAND_BENCHMARK_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build
include_directories(${PROJECT_SOURCE_DIR}/library/include) # hipcc workaround
include_directories(${PROJECT_BINARY_DIR}/library/include) # hipcc workaround

if(HIP_PLATFORM STREQUAL "nvcc")
    set(HIP_NVCC_FLAGS " ${HIP_NVCC_FLAGS} --expt-extended-lambda")
endif()

foreach(benchmark_src ${rocRAND_BENCHMARK_SRCS})
    get_filename_component(benchmark_name ${benchmark_src} NAME_WE)
    hip_add_executable(${benchmark_name} "${benchmark_src}")
    add_dependencies(${benchmark_name} rocrand ${ROCRAND_BENCHMARK_DEPENDENCIES})
    get_target_property(rocRAND_rpath rocrand BINARY_DIR)
    target_link_libraries(
        ${benchmark_name}
        -Wl,-rpath,${rocRAND_rpath} # workaround for HIP linker
        $<TARGET_FILE:rocrand> # rocRAND
    )
    if(HIP_PLATFORM STREQUAL "nvcc")
        target_link_libraries(${benchmark_name} "-lcurand")
    else()
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${benchmark_name} --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(
        ${benchmark_name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
    )
endforeach()
