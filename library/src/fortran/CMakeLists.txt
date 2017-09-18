# Fortran Wrapper

if(HIP_PLATFORM STREQUAL "nvcc")
    set(FORTRAN_ROCRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/rocrand_m.f90 ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_nvcc_m.f90)
    set(FORTRAN_HIPRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/hiprand_m.f90 ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_nvcc_m.f90)
else()
set(FORTRAN_ROCRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/rocrand_m.f90 ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_hcc_m.f90)
set(FORTRAN_HIPRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/hiprand_m.f90 ${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_hcc_m.f90)
endif()

set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/library/fortran)

# Rocrand Fortran Wrapper
add_library(rocrand_fortran SHARED "${FORTRAN_ROCRAND_SRCS}")
set_target_properties(rocrand_fortran PROPERTIES LINKER_LANGUAGE Fortran)
add_dependencies(rocrand_fortran rocrand)
get_target_property(rocRAND_rpath rocrand BINARY_DIR)
target_link_libraries(
    rocrand_fortran
    -Wl,-rpath,${rocRAND_rpath} # workaround for HIP linker
    $<TARGET_FILE:rocrand>
)

# Hiprand Fortran Wrapper
add_library(hiprand_fortran SHARED "${FORTRAN_HIPRAND_SRCS}")
set_target_properties(hiprand_fortran PROPERTIES LINKER_LANGUAGE Fortran)
add_dependencies(hiprand_fortran hiprand)
get_target_property(hipRAND_rpath hiprand BINARY_DIR)
target_link_libraries(
    hiprand_fortran
    -Wl,-rpath,${hipRAND_rpath} # workaround for HIP linker
    $<TARGET_FILE:hiprand>
)
