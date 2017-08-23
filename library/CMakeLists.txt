# rocRAND library

# Get sources
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
set(rocRAND_SRCS ${tmp})
file(GLOB tmp ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/*.cpp)
foreach(EXCLUDE ${tmp})
    list(REMOVE_ITEM rocRAND_SRCS "${EXCLUDE}")
endforeach()
set_source_files_properties(${rocRAND_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build library
include_directories(include)
hip_add_library(rocrand SHARED "${rocRAND_SRCS}")
set_target_properties(
    rocrand
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
        DEBUG_POSTFIX "-d"
)

# Install
# .so lib
install(
    TARGETS rocrand
    EXPORT rocrand-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/rocrand/lib
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)
# headers
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/rocrand/include
    FILES_MATCHING PATTERN "rocrand*"
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
# .cmake files
install(
    EXPORT rocrand-targets
    DESTINATION ${CMAKE_INSTALL_PREFIX}/rocrand/lib/rocrand/cmake
)
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/rocrand/include")
set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/rocrand/lib")
include(CMakePackageConfigHelpers)
configure_package_config_file(
    src/rocrand-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/rocrand-config.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/rocrand/cmake
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/rocrand-config-version.cmake
    VERSION ${rocrand_version}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/rocrand-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/rocrand-config-version.cmake
    DESTINATION ${LIB_INSTALL_DIR}/rocrand/cmake
)

# hipRAND library

# Get sources
file(GLOB hipRAND_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/hiprand/*_${HIP_PLATFORM}.cpp)
set_source_files_properties(${hipRAND_SRCS} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT TRUE)

# Build
hip_add_library(hiprand SHARED "${hipRAND_SRCS}")
if(HIP_PLATFORM STREQUAL "nvcc")
    target_link_libraries(hiprand "-lcurand")
else()
    add_dependencies(hiprand rocrand)
    target_link_libraries(
        hiprand
        -Wl,-rpath,${rocRAND_rpath} # workaround for HIP linker
        $<TARGET_FILE:rocrand>
    )
endif()
set_target_properties(
    hiprand
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/library"
)

# Install
# .so library
install(
    TARGETS hiprand
    EXPORT hiprand-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/hiprand/lib
    PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ GROUP_EXECUTE GROUP_READ WORLD_EXECUTE WORLD_READ
)
# headers
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_PREFIX}/hiprand/include
    FILES_MATCHING PATTERN "hiprand*"
    PERMISSIONS OWNER_WRITE OWNER_READ GROUP_READ WORLD_READ
)
# .cmake files
install(
    EXPORT hiprand-targets
    DESTINATION ${CMAKE_INSTALL_PREFIX}/hiprand/lib/hiprand/cmake
)
set(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/hiprand/include")
set(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/hiprand/lib")
include(CMakePackageConfigHelpers)
configure_package_config_file(
    src/hiprand-config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config.cmake
    INSTALL_DESTINATION ${LIB_INSTALL_DIR}/hiprand/cmake
    PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config-version.cmake
    VERSION ${hiprand_version}
    COMPATIBILITY SameMajorVersion
)
install(
    FILES
        ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/hiprand-config-version.cmake
    DESTINATION ${LIB_INSTALL_DIR}/hiprand/cmake
)
