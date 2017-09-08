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
include_directories(${PROJECT_SOURCE_DIR}/library/include) # hipcc workaround
include_directories(${PROJECT_BINARY_DIR}/library/include) # hipcc workaround
include_directories(${PROJECT_SOURCE_DIR}/library/src) # hipcc workaround
include_directories(${PROJECT_SOURCE_DIR}/test/crush) # hipcc workaround
include_directories(SYSTEM ${TestU01_INCLUDE_DIR}) # hipcc workaround

if(HIP_PLATFORM STREQUAL "nvcc")
    set(HIP_NVCC_FLAGS " ${HIP_NVCC_FLAGS} --expt-extended-lambda")
endif()

foreach(crush_test_src ${rocRAND_CRUSH_TEST_SRCS})
    get_filename_component(crush_test_name ${crush_test_src} NAME_WE)
    hip_add_executable(${crush_test_name} "${crush_test_src}")
    add_dependencies(${crush_test_name} rocrand ${ROCRAND_CRUSH_TEST_DEPENDENCIES})
    get_target_property(rocRAND_rpath rocrand BINARY_DIR)
    set(TestU01_rpath "${TestU01_LIBRARY_DIRS}")
    target_link_libraries(
        ${crush_test_name}
        -Wl,-rpath,${rocRAND_rpath}:${TestU01_rpath} # workaround for HIP linker
        $<TARGET_FILE:rocrand>
        ${TestU01_LIBRARIES}
    )
    if(HIP_PLATFORM STREQUAL "nvcc")
        target_link_libraries(${crush_test_name} "-lcurand")
    else()
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${crush_test_name} --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(
        ${crush_test_name}
        PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
endforeach()
