# Fortran Wrapper Tests

file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/fruit/*.f90)
set(FRUIT_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/fruit_helpers.f90)
set(FRUIT_HELPER_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*_rocrand.f90)
set(FORTRAN_ROCRAND_TEST ${tmp})

set(test_rocrand "test_rocrand_fortran_wrapper")

# Build unit test (rocrand)
add_executable(${test_rocrand} "${FRUIT_SRCS}" "${FRUIT_HELPER_SRCS}" "${FORTRAN_ROCRAND_TEST}")
target_include_directories(${test_rocrand}
    PRIVATE
        "${PROJECT_BINARY_DIR}/library/fortran"
)
target_link_libraries(${test_rocrand} rocrand_fortran)
set_target_properties(${test_rocrand}
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)

add_relative_test(${test_rocrand})
