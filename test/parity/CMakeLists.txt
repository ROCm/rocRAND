# MIT License
#
# Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Tests whether rocRAND and cuRAND generate the same stream of numbers
# when legacy ordering is used.

if(NOT BUILD_PARITY_TEST OR NOT USE_HIP_CPU)
    return()
endif()

# Compile the files that invoke each API separately, to avoid symbol namespacing errors.

# The curand part. We are compiling with a regular compiler, so we need to link the relevant cuda libraries.
add_library(test_rocrand_parity_curand OBJECT parity_curand.cpp)

if(CMAKE_VERSION VERSION_LESS 3.17)
    find_package(CUDA REQUIRED)
else()
    find_package(CUDAToolkit)
    set(CUDA_LIBRARIES CUDA::cudart)
    set(CUDA_curand_LIBRARY CUDA::curand)
    set(CUDA_INCLUDE_DIRS CUDAToolkit_INCLUDE_DIRS)
endif()

target_include_directories(test_rocrand_parity_curand PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(test_rocrand_parity_curand ${CUDA_LIBRARIES} ${CUDA_curand_LIBRARY})

# The rocrand part.
add_library(test_rocrand_parity_rocrand OBJECT parity_rocrand.cpp)
target_link_libraries(test_rocrand_parity_rocrand
    rocrand
    Threads::Threads
    hip_cpu_rt::hip_cpu_rt
)
if(STL_DEPENDS_ON_TBB)
    target_link_libraries(test_rocrand_parity_rocrand TBB::tbb)
endif()
target_include_directories(test_rocrand_parity_rocrand
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/library/src>
)

# The main test executable.
add_executable(test_rocrand_parity parity.cpp)
target_link_libraries(test_rocrand_parity
    test_rocrand_parity_curand
    test_rocrand_parity_rocrand
)
set_target_properties(test_rocrand_parity
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/test"
)

# add_relative_test(test_rocrand_parity test_rocrand_parity)
