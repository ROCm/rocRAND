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

#include <cuda_runtime.h>
#include <curand.h>

#include <iostream>

#define CUDA_CHECK(condition)                                  \
    {                                                          \
        cudaError_t error = condition;                         \
        if(error != cudaSuccess)                               \
        {                                                      \
            std::cout << "cuda error: " << error << std::endl; \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    }

#define CURAND_CHECK(condition)                                  \
    {                                                            \
        curandStatus_t error = condition;                        \
        if(error != CURAND_STATUS_SUCCESS)                       \
        {                                                        \
            std::cout << "curand error: " << error << std::endl; \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    }


void test_curand()
{
    constexpr size_t size = 1024;

    curandGenerator_t generator;
    CURAND_CHECK(curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetGeneratorOrdering(generator, CURAND_ORDERING_PSEUDO_LEGACY));

    unsigned int* data;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data), size * sizeof(unsigned int)));

    CURAND_CHECK(curandGenerate(generator, data, size));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(data));

    CURAND_CHECK(curandDestroyGenerator(generator));
}
