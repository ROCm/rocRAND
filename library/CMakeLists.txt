# hipRAND library

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(hipRAND_SRCS ${tmp})
set_source_files_properties(${hipRAND_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build library
include_directories(include)
hip_add_library(hipRAND SHARED "${hipRAND_SRCS}")
set_target_properties(
        hipRAND
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
    )
