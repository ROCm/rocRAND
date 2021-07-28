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
    if (WIN32 AND NOT DEFINED DLLS_COPIED)
      set(DLLS_COPIED "YES")
      set(DLLS_COPIED ${DLLS_COPIED} PARENT_SCOPE)
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
    )
endforeach()
