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
include_directories(SYSTEM ${Boost_INCLUDE_DIR}) # Boost
include_directories(${PROJECT_SOURCE_DIR}/library/include) # rocRAND

if(HIP_PLATFORM STREQUAL "nvcc")
    set(HIP_NVCC_FLAGS " ${HIP_NVCC_FLAGS} --expt-extended-lambda")
endif()

foreach(benchmark_src ${rocRAND_BENCHMARK_SRCS})
    get_filename_component(benchmark_name ${benchmark_src} NAME_WE)
    hip_add_executable(${benchmark_name} "${benchmark_src}")
    add_dependencies(${benchmark_name} rocRAND ${ROCRAND_BENCHMARK_DEPENDENCIES})
    get_target_property(rocRAND_rpath rocRAND BINARY_DIR)
    set(Boost_rpath "${Boost_LIBRARY_DIRS}")
    if("${Boost_rpath}" STREQUAL "")
        set(Boost_rpath "${PROJECT_SOURCE_DIR}/boost/lib")
    endif()
    target_link_libraries(
        ${benchmark_name}
        -Wl,-rpath,${rocRAND_rpath}:${Boost_rpath} # workaround for HIP linker
        $<TARGET_FILE:rocRAND> # rocRAND
        ${Boost_LIBRARIES} # Boost program_options
    )
    if(HIP_PLATFORM STREQUAL "nvcc")
        target_link_libraries( ${benchmark_name} "-lcurand")
    endif()
    set_target_properties(
        ${benchmark_name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
    )
endforeach()
