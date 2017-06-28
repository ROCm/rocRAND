# Benchmarks

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
set(hipRAND_TEST_SRCS ${tmp})
set_source_files_properties(${hipRAND_BENCHMARK_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build
include_directories(${PROJECT_SOURCE_DIR}/library/include)
foreach(test_src ${hipRAND_BENCHMARK_SRCS})
    get_filename_component(bin_name ${bin_src} NAME_WE)
    hip_add_executable(${bench_name} "${bin_src}")
    add_dependencies(${bench_name} hipRAND)
    get_target_property(hipRAND_rpath hipRAND BINARY_DIR)
    target_link_libraries(
        ${test_name}
        -Wl,-rpath,${hipRAND_rpath} # workaround for HIP linker
        $<TARGET_FILE:hipRAND>)
    set_target_properties(
        ${bench_name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/benchmark"
    )
endforeach()
