// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <gtest/gtest.h>
#include <stdexcept>
#include <stdio.h>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include <rng/generator_type.hpp>
#include <rng/lfsr113.hpp>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"

struct rocrand_lfsr113_prng_tests : public testing::TestWithParam<rocrand_ordering>
{
    rocrand_lfsr113 get_generator() const
    {
        rocrand_lfsr113 g;
        if(g.set_order(GetParam()) != ROCRAND_STATUS_SUCCESS)
        {
            throw std::runtime_error("Could not set ordering for generator");
        }
        return g;
    }
};

INSTANTIATE_TEST_SUITE_P(rocrand,
                         rocrand_lfsr113_prng_tests,
                         testing::Values(ROCRAND_ORDERING_PSEUDO_DEFAULT,
                                         ROCRAND_ORDERING_PSEUDO_DYNAMIC));

TEST_P(rocrand_lfsr113_prng_tests, uniform_uint_test)
{
    const size_t  size = 1313;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

    rocrand_lfsr113 g = get_generator();
    ROCRAND_CHECK(g.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long sum = 0;
    for(size_t i = 0; i < size; i++)
    {
        sum += host_data[i];
    }
    const unsigned int mean = sum / size;
    ASSERT_NEAR(mean, UINT_MAX / 2, UINT_MAX / 20);

    HIP_CHECK(hipFree(data));
}

TEST_P(rocrand_lfsr113_prng_tests, uniform_float_test)
{
    const size_t size = 1313;
    float*       data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(float) * size));

    rocrand_lfsr113 g = get_generator();
    ROCRAND_CHECK(g.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    float host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double sum = 0;
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_GT(host_data[i], 0.0f);
        ASSERT_LE(host_data[i], 1.0f);
        sum += host_data[i];
    }
    const float mean = sum / size;
    ASSERT_NEAR(mean, 0.5f, 0.05f);

    HIP_CHECK(hipFree(data));
}

TEST_P(rocrand_lfsr113_prng_tests, normal_float_test)
{
    const size_t size = 1313;
    float*       data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(float) * size));

    rocrand_lfsr113 g = get_generator();
    ROCRAND_CHECK(g.generate_normal(data, size, 2.0f, 5.0f));
    HIP_CHECK(hipDeviceSynchronize());

    float host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

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
    std = sqrt(std / size);

    EXPECT_NEAR(2.0f, mean, 0.4f); // 20%
    EXPECT_NEAR(5.0f, std, 1.0f); // 20%

    HIP_CHECK(hipFree(data));
}

TEST_P(rocrand_lfsr113_prng_tests, poisson_test)
{
    const size_t  size = 1313;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

    rocrand_lfsr113 g = get_generator();
    ROCRAND_CHECK(g.generate_poisson(data, size, 5.5));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;

    double var = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        double x = host_data[i] - mean;
        var += x * x;
    }
    var = var / size;

    EXPECT_NEAR(mean, 5.5, std::max(1.0, 5.5 * 1e-2));
    EXPECT_NEAR(var, 5.5, std::max(1.0, 5.5 * 1e-2));

    HIP_CHECK(hipFree(data));
}

// Check if the numbers generated by first generate() call are different from
// the numbers generated by the 2nd call (same generator)
TEST_P(rocrand_lfsr113_prng_tests, state_progress_test)
{
    // Device data
    const size_t  size = 1025;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

    // Generator
    rocrand_lfsr113 g0 = get_generator();

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data1[size];
    HIP_CHECK(hipMemcpy(host_data1, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data2[size];
    HIP_CHECK(hipMemcpy(host_data2, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    size_t same = 0;
    for(size_t i = 0; i < size; i++)
    {
        if(host_data1[i] == host_data2[i])
            same++;
    }
    // It may happen that numbers are the same, so we
    // just make sure that most of them are different.
    EXPECT_LT(same, static_cast<size_t>(0.01f * size));

    HIP_CHECK(hipFree(data));
}

// Checks if generators with the same seed and in the same state
// generate the same numbers
TEST_P(rocrand_lfsr113_prng_tests, same_seed_test)
{
    const uint4 seeds = {0, 2, 4, 6};

    // Device side data
    const size_t  size = 1024;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

    // Generators
    rocrand_lfsr113 g0 = get_generator(), g1 = get_generator();
    // Set same seeds
    g0.set_seed_uint4(seeds);
    g1.set_seed_uint4(seeds);

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int g0_host_data[size];
    HIP_CHECK(hipMemcpy(g0_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Generate using g1 and copy to host
    ROCRAND_CHECK(g1.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int g1_host_data[size];
    HIP_CHECK(hipMemcpy(g1_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Numbers generated using same generator with same
    // seeds should be the same
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(g0_host_data[i], g1_host_data[i]);
    }

    HIP_CHECK(hipFree(data));
}

// Checks if generators with the same seed and in the same state generate
// the same numbers
TEST_P(rocrand_lfsr113_prng_tests, different_seed_test)
{
    const unsigned long long seed0 = 5ULL;
    const unsigned long long seed1 = 10ULL;

    // Device side data
    const size_t  size = 1024;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

    // Generators
    rocrand_lfsr113 g0 = get_generator(), g1 = get_generator();
    // Set different seeds
    g0.set_seed(seed0);
    g1.set_seed(seed1);

    const uint4 get_seed0 = g0.get_seed_uint4();
    const uint4 get_seed1 = g1.get_seed_uint4();

    ASSERT_NE(get_seed0.x, get_seed1.x);

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int g0_host_data[size];
    HIP_CHECK(hipMemcpy(g0_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Generate using g1 and copy to host
    ROCRAND_CHECK(g1.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int g1_host_data[size];
    HIP_CHECK(hipMemcpy(g1_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    size_t same = 0;
    for(size_t i = 0; i < size; i++)
    {
        if(g1_host_data[i] == g0_host_data[i])
            same++;
    }
    // It may happen that numbers are the same, so we
    // just make sure that most of them are different.
    EXPECT_LT(same, static_cast<size_t>(0.01f * size));

    HIP_CHECK(hipFree(data));
}

// Checks if generators with the same seed and in the same state generate
// the same numbers
TEST_P(rocrand_lfsr113_prng_tests, different_seed_uint4_test)
{
    const uint4 seeds0[] = {
        { 0,  2,  4,  6},
        { 0,  2,  4,  6},
        {11, 12, 13, 14}
    };
    const uint4 seeds1[] = {
        { 3,  5, 22, 1000},
        {10, 30, 60,  900},
        {12, 13, 14,  150}
    };

    for(unsigned int i = 0; i < 3; i++)
    {
        const uint4 seed0 = seeds0[i];
        const uint4 seed1 = seeds1[i];

        // Device side data
        const size_t  size = 1024;
        unsigned int* data;
        HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

        // Generators
        rocrand_lfsr113 g0 = get_generator(), g1 = get_generator();
        // Set different seeds
        g0.set_seed_uint4(seed0);
        g1.set_seed_uint4(seed1);

        const uint4 get_seed0 = g0.get_seed_uint4();
        const uint4 get_seed1 = g1.get_seed_uint4();

        ASSERT_NE(get_seed0.x, get_seed1.x);
        ASSERT_NE(get_seed0.y, get_seed1.y);
        ASSERT_NE(get_seed0.z, get_seed1.z);
        ASSERT_NE(get_seed0.w, get_seed1.w);

        // Generate using g0 and copy to host
        ROCRAND_CHECK(g0.generate(data, size));
        HIP_CHECK(hipDeviceSynchronize());

        unsigned int g0_host_data[size];
        HIP_CHECK(
            hipMemcpy(g0_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        // Generate using g1 and copy to host
        ROCRAND_CHECK(g1.generate(data, size));
        HIP_CHECK(hipDeviceSynchronize());

        unsigned int g1_host_data[size];
        HIP_CHECK(
            hipMemcpy(g1_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        size_t same = 0;
        for(size_t i = 0; i < size; i++)
        {
            if(g1_host_data[i] == g0_host_data[i])
                same++;
        }
        // It may happen that numbers are the same, so we
        // just make sure that most of them are different.
        EXPECT_LT(same, static_cast<size_t>(0.01f * size));

        HIP_CHECK(hipFree(data));
    }
}
