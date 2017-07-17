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

TEST(rocrand_mrg32k3a_prng_tests, uniform_uint_test)
{
    const size_t size = 1313;
    unsigned int * data;
    hipMalloc(&data, sizeof(unsigned int) * size);

    rocrand_mrg32k3a g;
    g.generate(data, size);
    hipDeviceSynchronize();

    unsigned int host_data[size];
    hipMemcpy(host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    for(size_t i = 0; i < size; i++)
    {
        const unsigned int max = UINT_MAX;
        ASSERT_GE(host_data[i], 0);
        ASSERT_LE(host_data[i], max);
    }
}

TEST(rocrand_mrg32k3a_prng_tests, uniform_float_test)
{
    const size_t size = 1313;
    float * data;
    hipMalloc(&data, sizeof(float) * size);

    rocrand_mrg32k3a g;
    g.generate(data, size);
    hipDeviceSynchronize();

    float host_data[size];
    hipMemcpy(host_data, data, sizeof(float) * size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_GT(host_data[i], 0.0f);
        ASSERT_LE(host_data[i], 1.0f);
    }
}

TEST(rocrand_mrg32k3a_prng_tests, normal_float_test)
{
    const size_t size = 1313;
    float * data;
    hipMalloc(&data, sizeof(float) * size);

    rocrand_mrg32k3a g;
    g.generate_normal(data, size, 5.0f, 2.0f);
    hipDeviceSynchronize();

    float host_data[size];
    hipMemcpy(host_data, data, sizeof(float) * size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    
    float mean = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;
    
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(host_data[i] - mean, 2);
    }
    std = std::sqrt(std / size);
    
    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%
}

TEST(rocrand_mrg32k3a_prng_tests, log_normal_float_test)
{
    const size_t size = 1313;
    float * data;
    hipMalloc(&data, sizeof(float) * size);

    rocrand_mrg32k3a g;
    g.generate_log_normal(data, size, 2.0f, 5.0f);
    hipDeviceSynchronize();

    float host_data[size];
    hipMemcpy(host_data, data, sizeof(float) * size, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();
    
    float mean = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;
    
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(host_data[i] - mean, 2);
    }
    std = std::sqrt(std / size);
    
    float logmean = logf(mean * mean / sqrtf(std * std + mean * mean));
    float logstd = sqrtf(logf(std * std / mean / mean + 1.0f));
    
    EXPECT_NEAR(5.0f, logmean, 1.0f);
    EXPECT_NEAR(2.0f, logstd, 1.0f);
}
