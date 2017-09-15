# Fortran Wrapper Tests
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/fruit/*.f90)
set(FRUIT_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.F90)
set(FRUIT_HELPER_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/hip_fortran/*.f90)
set(HIP_FORTRAN_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/*.f90)
set(FORTRAN_ROCRAND_TEST ${tmp})
set(FORTRAN_ROCRAND_CPP ${CMAKE_CURRENT_SOURCE_DIR}/hip_fortran/hip_fortran.cpp)
set_source_files_properties(${FORTRAN_ROCRAND_CPP} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

hip_add_library(hip_fortran SHARED "${FORTRAN_ROCRAND_CPP}")
include_directories(${PROJECT_BINARY_DIR}/library)
set(test_name "test_rocrand_fortran_wrapper")
add_executable(${test_name} "${FRUIT_SRCS}" "${FRUIT_HELPER_SRCS}" "${HIP_FORTRAN_SRCS}" "${FORTRAN_ROCRAND_TEST}")
add_dependencies(${test_name} rocrand rocrand_fortran hip_fortran)
get_target_property(rocRAND_fortran_rpath rocrand_fortran BINARY_DIR)
get_target_property(rocRAND_rpath rocrand BINARY_DIR)
get_target_property(hip_fortran_rpath hip_fortran BINARY_DIR)
target_link_libraries(
    ${test_name}
    -Wl,-rpath,${rocRAND_fortran_rpath}:${rocRAND_rpath}:${hip_fortran_rpath} # workaround for HIP linker
    $<TARGET_FILE:rocrand_fortran>
    $<TARGET_FILE:rocrand>
    $<TARGET_FILE:hip_fortran>
)
if(HIP_PLATFORM STREQUAL "nvcc")
    target_link_libraries(${test_name} "-lcuda -lcudart")
endif()
set_target_properties(
    ${test_name} 
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)
add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)