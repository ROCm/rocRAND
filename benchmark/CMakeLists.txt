# Benchmarks

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rocrand*.cpp)
set(rocRAND_BENCHMARK_SRCS ${tmp})
if(HIP_COMPILER STREQUAL "nvcc")
    file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/benchmark_curand*.cpp)
    set(rocRAND_BENCHMARK_SRCS ${rocRAND_BENCHMARK_SRCS} ${tmp})
endif()

# Build benchmarks
foreach(benchmark_src ${rocRAND_BENCHMARK_SRCS})
    get_filename_component(BENCHMARK_TARGET ${benchmark_src} NAME_WE)
    add_executable(${BENCHMARK_TARGET} ${benchmark_src})
    # nvcc/CUDA
    if(HIP_COMPILER STREQUAL "nvcc")
      set_property(TARGET ${BENCHMARK_TARGET} PROPERTY CUDA_STANDARD 14)
      set_source_files_properties(${benchmark_src} PROPERTIES LANGUAGE CUDA)
      target_compile_options(${BENCHMARK_TARGET}
        PRIVATE
          $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
      )
    endif()
    if(HIP_COMPILER STREQUAL "nvcc")
        target_link_libraries(${BENCHMARK_TARGET}
            rocrand
            ${CUDA_curand_LIBRARY}
        )
    else()
        target_link_libraries(${BENCHMARK_TARGET}
            rocrand
            hip::device
        )
    endif()
    set_target_properties(${BENCHMARK_TARGET}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
    )
endforeach()
