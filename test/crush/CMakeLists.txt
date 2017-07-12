# Crush Tests

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*rocrand*.cpp)
set(rocRAND_CRUSH_TEST_SRCS ${tmp})
if(HIP_PLATFORM STREQUAL "nvcc")
    file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*curand*.cpp)
    set(rocRAND_CRUSH_TEST_SRCS ${rocRAND_CRUSH_TEST_SRCS} ${tmp})
endif()
set_source_files_properties(${rocRAND_CRUSH_TEST_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build
include_directories(SYSTEM ${Boost_INCLUDE_DIRS}) # Boost
include_directories(${PROJECT_SOURCE_DIR}/library/include)
include_directories(${PROJECT_SOURCE_DIR}/library/src)
include_directories(${PROJECT_SOURCE_DIR}/test/crush)
include_directories(${TESTU01_INCLUDE_DIRS})

foreach(crush_test_src ${rocRAND_CRUSH_TEST_SRCS})
    get_filename_component(crush_test_name ${crush_test_src} NAME_WE)
    hip_add_executable(${crush_test_name} "${crush_test_src}")
    add_dependencies(${crush_test_name} rocRAND)
    get_target_property(rocRAND_rpath rocRAND BINARY_DIR)
    target_link_libraries(
        ${crush_test_name}
        -Wl,-rpath,${TESTU01_LIBRARY_DIRS}:${rocRAND_rpath} # workaround for HIP linker
        $<TARGET_FILE:rocRAND> ${testu01} ${probdist} ${mylib}
        ${Boost_LIBRARIES} # Boost program_options
    )
    if(HIP_PLATFORM STREQUAL "nvcc")
        target_link_libraries( ${crush_test_name} "-lcurand")
    endif()
    set_target_properties(
        ${crush_test_name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
endforeach()
