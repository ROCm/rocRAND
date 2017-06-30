# rocRAND library

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(rocRAND_SRCS ${tmp})
set_source_files_properties(${rocRAND_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build library
include_directories(include)
hip_add_library(rocRAND SHARED "${rocRAND_SRCS}")
set_target_properties(
        rocRAND
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
    )
