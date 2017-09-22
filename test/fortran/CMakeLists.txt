# Fortran Wrapper Tests

file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/fruit/*.f90)
set(FRUIT_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/fruit_helpers.f90)
set(FRUIT_HELPER_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*_rocrand.f90)
set(FORTRAN_ROCRAND_TEST ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*_hiprand.f90)
set(FORTRAN_HIPRAND_TEST ${tmp})

set(test_rocrand "test_rocrand_fortran_wrapper")
set(test_hiprand "test_hiprand_fortran_wrapper")

# Build unit test (rocrand)
add_executable(${test_rocrand} "${FRUIT_SRCS}" "${FRUIT_HELPER_SRCS}" "${FORTRAN_ROCRAND_TEST}")
add_dependencies(${test_rocrand} rocrand rocrand_fortran)
target_include_directories(
    ${test_rocrand}
    PRIVATE
        "${PROJECT_BINARY_DIR}/library/fortran"
)
target_link_libraries(
    ${test_rocrand}
        $<TARGET_FILE:rocrand_fortran>
        $<TARGET_FILE:rocrand>
)
if(HIP_PLATFORM STREQUAL "nvcc")
    target_link_libraries(
        ${test_rocrand}
        "-lcuda -lcudart -lcurand"
    )
else()
    target_link_libraries(
        ${test_rocrand}
        "-L${HIP_ROOT_DIR}/lib -lhip_hcc -Wl,-rpath,${HIP_ROOT_DIR}/lib"
    )
endif()
set_target_properties(
    ${test_rocrand}
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)

# Build unit test (hiprand)
add_executable(${test_hiprand} "${FRUIT_SRCS}" "${FRUIT_HELPER_SRCS}" "${FORTRAN_HIPRAND_TEST}")
add_dependencies(${test_hiprand} hiprand hiprand_fortran)
target_include_directories(
    ${test_hiprand}
    PRIVATE
        "${PROJECT_BINARY_DIR}/library/fortran"
)
target_link_libraries(
    ${test_hiprand}
        $<TARGET_FILE:hiprand_fortran>
        $<TARGET_FILE:hiprand>
)
if(HIP_PLATFORM STREQUAL "nvcc")
    target_link_libraries(
        ${test_hiprand}
        "-lcuda -lcudart -lcurand"
    )
else()
    target_link_libraries(
        ${test_hiprand}
        $<TARGET_FILE:rocrand>
        "-L${HIP_ROOT_DIR}/lib -lhip_hcc -Wl,-rpath,${HIP_ROOT_DIR}/lib"
    )
endif()
set_target_properties(
    ${test_hiprand}
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)


add_test(NAME ${test_rocrand} COMMAND $<TARGET_FILE:${test_rocrand}>)
add_test(NAME ${test_hiprand} COMMAND $<TARGET_FILE:${test_hiprand}>)
