# MIT License
#
# Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

set(CMAKE_CXX_STANDARD 17)

add_custom_target(tools)

function(rocrand_add_tool TARGET_NAME)
    add_executable(${TARGET_NAME} ${ARGN})
    add_dependencies(tools ${TARGET_NAME})
endfunction()

rocrand_add_tool(bin2typed bin2typed.cpp)
rocrand_add_tool(mrg31k3p_precomputed_generator mrg31k3p_precomputed_generator.cpp)
rocrand_add_tool(mrg32k3a_precomputed_generator mrg32k3a_precomputed_generator.cpp)
rocrand_add_tool(mt19937_precomputed_generator mt19937_precomputed_generator.cpp)
rocrand_add_tool(scrambled_sobol32_constants_generator scrambled_sobol32_constants_generator.cpp)
rocrand_add_tool(scrambled_sobol32_direction_vector_generator
    scrambled_sobol32_direction_vector_generator.cpp)
rocrand_add_tool(scrambled_sobol64_constants_generator scrambled_sobol64_constants_generator.cpp)
rocrand_add_tool(scrambled_sobol64_direction_vector_generator
    scrambled_sobol64_direction_vector_generator.cpp)
rocrand_add_tool(sobol32_direction_vector_generator sobol32_direction_vector_generator.cpp)
rocrand_add_tool(sobol64_direction_vector_generator sobol64_direction_vector_generator.cpp)
rocrand_add_tool(xorwow_precomputed_generator xorwow_precomputed_generator.cpp)

