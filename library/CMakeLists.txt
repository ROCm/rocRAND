# rocRAND library

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(rocRAND_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/*.cpp)
foreach(EXCLUDE ${tmp})
    list(REMOVE_ITEM rocRAND_SRCS "${EXCLUDE}")
endforeach()
set_source_files_properties(${rocRAND_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build library
include_directories(include)
hip_add_library(rocRAND SHARED "${rocRAND_SRCS}")
set_target_properties(
    rocRAND
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
)

# hipRAND library

# Get sources
file(GLOB hipRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/*_${HIP_PLATFORM}.cpp)
set_source_files_properties(${hipRAND_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build
hip_add_library(hipRAND SHARED "${hipRAND_SRCS}")
if(HIP_PLATFORM STREQUAL "nvcc")
    target_link_libraries(hipRAND "-lcurand")
else()
    target_link_libraries(
        hipRAND
        -Wl,-rpath,${rocRAND_rpath} # workaround for HIP linker
        $<TARGET_FILE:rocRAND>
    )
endif()
set_target_properties(
    hipRAND
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
)
