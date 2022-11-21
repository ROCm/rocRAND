// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "parity.hpp"

#include <rocrand/rocrand.h>
#include <hip/hip_runtime.h>

#include <iostream>

#define HIP_CHECK(condition)                                  \
    {                                                         \
        hipError_t error = condition;                         \
        if(error != hipSuccess){                              \
            std::cout << "hip error: " << error << std::endl; \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    }

#define ROCRAND_CHECK(condition)                                   \
    {                                                              \
        rocrand_status status = condition;                         \
        if(status != ROCRAND_STATUS_SUCCESS) {                     \
            std::cout << "rocrand error: " << status << std::endl; \
            exit(EXIT_FAILURE);                                    \
        }                                                          \
    }

void test_rocrand()
{
    constexpr size_t size = 1024;

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, ROCRAND_RNG_PSEUDO_DEFAULT));
    ROCRAND_CHECK(rocrand_set_ordering(generator, ROCRAND_ORDERING_PSEUDO_LEGACY));

    unsigned int* data;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), size * sizeof(unsigned int)));

    ROCRAND_CHECK(rocrand_generate(generator, data, size));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));

    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}
