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

add_executable(xorwow_precomputed_generator xorwow_precomputed_generator.cpp)
add_executable(sobol32_direction_vector_generator sobol32_direction_vector_generator.cpp)
add_executable(scrambled_sobol32_direction_vector_generator scrambled_sobol32_direction_vector_generator.cpp)
add_executable(scrambled_sobol32_constants_generator scrambled_sobol32_constants_generator.cpp)
add_executable(sobol64_direction_vector_generator sobol64_direction_vector_generator.cpp)
add_executable(scrambled_sobol64_direction_vector_generator scrambled_sobol64_direction_vector_generator.cpp)
add_executable(scrambled_sobol64_constants_generator scrambled_sobol64_constants_generator.cpp)
add_executable(mrg31k3p_precomputed_generator mrg31k3p_precomputed_generator.cpp)
add_executable(mrg32k3a_precomputed_generator mrg32k3a_precomputed_generator.cpp)
add_executable(mt19937_precomputed_generator mt19937_precomputed_generator.cpp)
