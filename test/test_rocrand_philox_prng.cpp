// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <stdio.h>
#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <rocrand.h>

#include <rng/generator_type.hpp>
#include <rng/generators.hpp>

TEST(rocrand_philox_prng_tests, example_philox_rng_test)
{
    constexpr size_t size = 10;
    unsigned int * data;
    hipMalloc(&data, sizeof(unsigned int) * size);

    rocrand_philox4x32_10 g;
    g.generate(data, size);
    hipDeviceSynchronize();

    unsigned int host_data[size];
    hipMemcpy(host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    for(size_t i = 0; i < size; i++)
        EXPECT_EQ(43210, host_data[i]);
}

TEST(rocrand_philox_prng_tests, normal_philox_rng_test)
{
    constexpr size_t size = 10;
    float * data;
    hipMalloc(&data, sizeof(float) * size);

    rocrand_philox4x32_10 g;
    g.generate_normal(data, size);
    hipDeviceSynchronize();

    float host_data[size];
    hipMemcpy(host_data, data, sizeof(float) * size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    for(size_t i = 0; i < size; i++)
        EXPECT_NEAR(0.0003, host_data[i], 0.0001);
}
