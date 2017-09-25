# Fortran Wrapper

if(HIP_PLATFORM STREQUAL "nvcc")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_nvcc_m.f90"
        "${PROJECT_BINARY_DIR}/src/fortran/hip_m.f90"
        COPYONLY
    )
else()
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/hip/hip_hcc_m.f90"
        "${PROJECT_BINARY_DIR}/src/fortran/hip_m.f90"
        COPYONLY
    )
endif()

set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/library/fortran)

# rocRAND Fortran wrapper
set(FORTRAN_ROCRAND_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/rocrand_m.f90
    ${PROJECT_BINARY_DIR}/src/fortran/hip_m.f90
)
if(BUILD_FORTRAN_WRAPPER)
    add_library(rocrand_fortran STATIC "${FORTRAN_ROCRAND_SRCS}")
    set_target_properties(rocrand_fortran PROPERTIES LINKER_LANGUAGE Fortran)
    add_dependencies(rocrand_fortran rocrand)
    target_link_libraries(rocrand_fortran
        PUBLIC
            rocrand
    )
endif()

# install Fortran sources
install(
    FILES ${FORTRAN_ROCRAND_SRCS}
    DESTINATION rocrand/src/fortran
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)

# hipRAND Fortran wrapper
set(FORTRAN_HIPRAND_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/hiprand_m.f90
    ${PROJECT_BINARY_DIR}/src/fortran/hip_m.f90
)
if(BUILD_FORTRAN_WRAPPER)
    add_library(hiprand_fortran STATIC "${FORTRAN_HIPRAND_SRCS}")
    set_target_properties(hiprand_fortran PROPERTIES LINKER_LANGUAGE Fortran)
    add_dependencies(hiprand_fortran hiprand)
    target_link_libraries(hiprand_fortran
        PUBLIC
            hiprand
    )
endif()

# install Fortran sources
install(
    FILES ${FORTRAN_HIPRAND_SRCS}
    DESTINATION hiprand/src/fortran
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
