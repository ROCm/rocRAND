# Crush Tests

# Get sources
file(GLOB rocRAND_CRUSH_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*rocrand*.cpp)
if(HIP_PLATFORM STREQUAL "nvcc")
    file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*curand*.cpp)
    set(rocRAND_CRUSH_TEST_SRCS ${rocRAND_CRUSH_TEST_SRCS} ${tmp})
endif()

# Use CUDA_INCLUDE_DIRECTORIES to include required dirs
# for nvcc if cmake version is less than 3.9.3
if((HIP_PLATFORM STREQUAL "nvcc") AND (CMAKE_VERSION VERSION_LESS "3.9.3"))
    CUDA_INCLUDE_DIRECTORIES(
        "${PROJECT_BINARY_DIR}/library/include/"
        "${PROJECT_SOURCE_DIR}/library/include/"
        ${TestU01_INCLUDE_DIR}
    )
endif()

# Build tests
foreach(crush_test_src ${rocRAND_CRUSH_TEST_SRCS})
    get_filename_component(crush_test_name ${crush_test_src} NAME_WE)
    # nvcc/CUDA
    if(HIP_PLATFORM STREQUAL "nvcc")
        set_source_files_properties(${crush_test_src}
            PROPERTIES
                CUDA_SOURCE_PROPERTY_FORMAT OBJ
        )
        CUDA_ADD_EXECUTABLE(${crush_test_name}
            ${crush_test_src}
            OPTIONS --expt-extended-lambda
        )
    # hcc/ROCm
    else()
        add_executable(${crush_test_name} ${crush_test_src})
    endif()
    # TestU01 include dirs
    target_include_directories(${crush_test_name} SYSTEM
        PUBLIC
            ${TestU01_INCLUDE_DIR}
    )
    if(HIP_PLATFORM STREQUAL "nvcc")
        target_link_libraries(${crush_test_name}
            rocrand
            ${CUDA_curand_LIBRARY}
            ${TestU01_LIBRARIES}
        )
    else()
        # Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
        if(TARGET hip::device)
            target_link_libraries(${crush_test_name}
                rocrand
                hip::device
                ${TestU01_LIBRARIES}
            )
        else()
            target_link_libraries(${crush_test_name}
                rocrand
                hip::hip_hcc hip::hip_device
                ${TestU01_LIBRARIES}
            )
        endif()
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${crush_test_name} --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(${crush_test_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
endforeach()
