# Tests

# Get rocRAND tests source files
file(GLOB rocRAND_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/test_hiprand*.cpp)
foreach(to_exclude ${tmp})
    list(REMOVE_ITEM rocRAND_TEST_SRCS "${to_exclude}")
endforeach()

# Build rocRAND tests
foreach(test_src ${rocRAND_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    # nvcc/CUDA
    if(HIP_COMPILER STREQUAL "nvcc")
      set_source_files_properties(${test_name}
        PROPERTIES
          LANGUAGE CUDA
          CUDA_STANDARD 14
      )
      set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()
    target_include_directories(${test_name}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/library/src
    )
    if(TARGET GTest::GTest)
        target_link_libraries(${test_name}
            GTest::GTest
            GTest::Main
        )
    else()
        target_link_libraries(${test_name}
            GTest::gtest
            GTest::gtest_main
        )
    endif()
    target_link_libraries(${test_name}
        rocrand
    )
    if(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
        # Remove this check when we no longer build with older rocm stack(ie < 1.8.2)
        if(TARGET hip::device)
            target_link_libraries(${test_name} hip::device)
        else()
            target_link_libraries(${test_name} hip::device)
        endif()
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

# Get hipRAND tests source files
file(GLOB hipRAND_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/test_hiprand*.cpp)

# Build hipRAND tests
foreach(test_src ${hipRAND_TEST_SRCS})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    # nvcc/CUDA
    if(HIP_COMPILER STREQUAL "nvcc")
      set_source_files_properties(${test_name}
        PROPERTIES
          LANGUAGE CUDA
          CUDA_STANDARD 14
      )
      set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    endif()
    target_include_directories(${test_name}
        PUBLIC
            ${PROJECT_SOURCE_DIR}/library/src
    )
    if(TARGET GTest::GTest)
        target_link_libraries(${test_name}
            GTest::GTest
            GTest::Main
        )
    else()
        target_link_libraries(${test_name}
            GTest::gtest
            GTest::gtest_main
        )
    endif()
    if(HIP_COMPILER  STREQUAL "nvcc")
        target_link_libraries(${test_name}
            hiprand
            ${CUDA_curand_LIBRARY}
        )
    else()
        target_link_libraries(${test_name}
            hiprand
            rocrand
            hip::device
        )
    endif()
    set_target_properties(${test_name}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
    )
    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
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
        add_custom_command( TARGET ${test_name} POST_BUILD COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${file_i} ${PROJECT_BINARY_DIR}/test )
      endforeach( file_i )
    endif()       
endforeach()
# Checks for simple linkage problems
add_subdirectory(linkage)
