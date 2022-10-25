# Benchmarks

function(add_rocrand_benchmark benchmark_src legacy)
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
      target_link_libraries(${BENCHMARK_TARGET} ${CUDA_curand_LIBRARY})
  else()
      target_link_libraries(${BENCHMARK_TARGET} hip::device)
  endif()
  target_link_libraries(${BENCHMARK_TARGET} rocrand)
  if(NOT legacy)
    target_link_libraries(${BENCHMARK_TARGET} benchmark::benchmark)
  endif()

  set_target_properties(${BENCHMARK_TARGET}
      PROPERTIES
          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
  )
  rocm_install(TARGETS ${BENCHMARK_TARGET} COMPONENT benchmarks)
  if (WIN32 AND NOT DEFINED DLLS_COPIED_BENCHMARKS)
    set(DLLS_COPIED_BENCHMARKS "YES")
    set(DLLS_COPIED_BENCHMARKS ${DLLS_COPIED_BENCHMARKS} PARENT_SCOPE)
    # for now adding in all .dll as dependency chain is not cmake based on win32
    file( GLOB third_party_dlls
    LIST_DIRECTORIES ON
    CONFIGURE_DEPENDS
    ${HIP_DIR}/bin/*.dll
    ${CMAKE_SOURCE_DIR}/rtest.*
    )
    foreach( file_i ${third_party_dlls})
      add_custom_command( TARGET ${BENCHMARK_TARGET} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/benchmark )
    endforeach( file_i )
  endif()
endfunction()

add_rocrand_benchmark("benchmark_rocrand_host_api.cpp" OFF)
add_rocrand_benchmark("benchmark_rocrand_device_api.cpp" OFF)
if(HIP_COMPILER STREQUAL "nvcc")
  add_rocrand_benchmark("benchmark_curand_host_api.cpp" OFF)
endif()

if(BUILD_LEGACY_BENCHMARK)
  add_rocrand_benchmark("benchmark_rocrand_generate.cpp" ON)
  add_rocrand_benchmark("benchmark_rocrand_kernel.cpp"   ON)
  if(HIP_COMPILER STREQUAL "nvcc")
    add_rocrand_benchmark("benchmark_curand_generate.cpp" ON)
    add_rocrand_benchmark("benchmark_curand_kernel.cpp"   ON)
  endif()
endif()

