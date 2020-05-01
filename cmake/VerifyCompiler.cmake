# MIT License
#
# Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.
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

find_package(hip REQUIRED CONFIG PATHS /opt/rocm)
message(STATUS "HIP_COMPILER=${HIP_COMPILER}")
message(STATUS "HIP_RUNTIME=${HIP_RUNTIME}")

if(HIP_COMPILER STREQUAL "nvcc")
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
        include(SetupNVCC)
    else()
        message(WARNING "On CUDA platform 'g++' is recommended C++ compiler.")
    endif()
elseif(HIP_COMPILER STREQUAL "hcc" OR HIP_COMPILER STREQUAL "clang")
    if(NOT (CMAKE_CXX_COMPILER MATCHES ".*/hcc$" OR CMAKE_CXX_COMPILER MATCHES ".*/hipcc$"))
        message(FATAL_ERROR "On ROCm platform 'hcc' or 'clang' must be used as C++ compiler.")
    elseif(CXX_VERSION_STRING MATCHES "clang")
        list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
    else()
        list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hcc /opt/rocm/hip)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
    endif()
else()
    message(FATAL_ERROR "HIP_COMPILER must be 'hcc' or 'clang' (AMD ROCm platform) or `nvcc` (NVIDIA CUDA platform).")
endif()
