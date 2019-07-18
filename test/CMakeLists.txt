# Tests

# Get rocRAND tests source files
file(GLOB rocRAND_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/test_hiprand*.cpp)
foreach(to_exclude ${tmp})
    list(REMOVE_ITEM rocRAND_TEST_SRCS "${to_exclude}")
endforeach()

# Use CUDA_INCLUDE_DIRECTORIES to include required dirs
# for nvcc if cmake version is less than 3.9.3
if((HIP_PLATFORM STREQUAL "nvcc") AND (CMAKE_VERSION VERSION_LESS "3.9.3"))
    CUDA_INCLUDE_DIRECTORIES(
        "${PROJECT_BINARY_DIR}/library/include/"
        "${PROJECT_SOURCE_DIR}/library/include/"
        "${PROJECT_SOURCE_DIR}/library/src"
        ${GTEST_INCLUDE_DIRS}
    )
endif()

# Build rocRAND tests
foreach(test_src ${rocRAND_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    # nvcc/CUDA
    if(HIP_PLATFORM STREQUAL "nvcc")
        set_source_files_properties(${test_src}
            PROPERTIES
                CUDA_SOURCE_PROPERTY_FORMAT OBJ
        )
        CUDA_ADD_EXECUTABLE(${test_name} ${test_src})
    # hcc/ROCm
    else()
        add_executable(${test_name} ${test_src})
    endif()
    # GTEST include dirs
    target_include_directories(${test_name} SYSTEM
        PUBLIC
            ${GTEST_INCLUDE_DIRS}
    )
    target_include_directories(${test_name}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/library/src
    )
    target_link_libraries(${test_name}
        rocrand
        ${GTEST_BOTH_LIBRARIES}
    )
    if(HIP_PLATFORM STREQUAL "hcc")
        # Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
        if(TARGET hip::device)
            target_link_libraries(${test_name} hip::device)
        else()
            target_link_libraries(${test_name} hip::hip_hcc hip::hip_device)
        endif()
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${test_name} --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(
        ${test_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()

# Fortran Wrapper Tests
if(BUILD_FORTRAN_WRAPPER)
    add_subdirectory(fortran)
endif()

# Crush Tests
if(BUILD_CRUSH_TEST)
    add_subdirectory(crush)
endif()

# Get hipRAND tests source files
file(GLOB hipRAND_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_hiprand*.cpp)

# Build hipRAND tests
foreach(test_src ${hipRAND_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    # nvcc/CUDA
    if(HIP_PLATFORM STREQUAL "nvcc")
        set_source_files_properties(${test_src}
            PROPERTIES
                CUDA_SOURCE_PROPERTY_FORMAT OBJ
        )
        CUDA_ADD_EXECUTABLE(${test_name} ${test_src})
    # hcc/ROCm
    else()
        add_executable(${test_name} ${test_src})
    endif()
    # GTEST include dirs
    target_include_directories(${test_name} SYSTEM
        PUBLIC
            ${GTEST_INCLUDE_DIRS}
    )
    target_include_directories(${test_name}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/library/src
    )
    if(HIP_PLATFORM STREQUAL "nvcc")
        target_link_libraries(${test_name}
            hiprand
            ${CUDA_curand_LIBRARY}
            ${GTEST_BOTH_LIBRARIES}
        )
    else()
        target_link_libraries(${test_name}
            hiprand
            rocrand
            ${GTEST_BOTH_LIBRARIES}
        )
        # Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
        if(TARGET hip::device)
            target_link_libraries(${test_name} hip::device)
        else()
            target_link_libraries(${test_name} hip::hip_hcc hip::hip_device)
        endif()
        foreach(amdgpu_target ${AMDGPU_TARGETS})
            target_link_libraries(${test_name} --amdgpu-target=${amdgpu_target})
        endforeach()
    endif()
    set_target_properties(${test_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()

# Checks for simple linkage problems
add_subdirectory(linkage)
