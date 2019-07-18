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

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)
#define ROCRAND_CHECK(state) ASSERT_EQ(state, ROCRAND_STATUS_SUCCESS)

using rocrand_device::detail::mad_u64_u32;

__global__
void mad_u64_u32_kernel(const unsigned int * x,
                        const unsigned int * y,
                        const unsigned long long * z,
                        unsigned long long * r)
{
    r[0] = mad_u64_u32(x[0], y[0], z[0]);
    r[1] = mad_u64_u32(x[1], y[1], z[1]);
    r[2] = mad_u64_u32(x[2], y[2], z[2]);
    r[3] = mad_u64_u32(x[3], y[3], z[3]);
    r[4] = mad_u64_u32(1403580, y[4], 5ULL);
    r[5] = mad_u64_u32(x[5], 1370589, 0ULL);
    r[6] = mad_u64_u32(0xFFFFFFFF, 0x87654321, 0x1234567890123456ULL);
    r[7] = mad_u64_u32(23, 45, 67ULL);
}

TEST(rocrand_mrg32k3a_prng_tests, mad_u64_u32_test)
{
    const size_t size = 8;

    unsigned int * x;
    unsigned int * y;
    unsigned long long * z;
    unsigned long long * r;
    HIP_CHECK(hipMalloc((void **)&x, size * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void **)&y, size * sizeof(unsigned int)));
    HIP_CHECK(hipMalloc((void **)&z, size * sizeof(unsigned long long)));
    HIP_CHECK(hipMalloc((void **)&r, size * sizeof(unsigned long long)));

    unsigned int h_x[size];
    unsigned int h_y[size];
    unsigned long long h_z[size];

    h_x[0] = 3492343451; h_y[0] = 1234; h_z[0] = 1231314234234265ULL;
    h_x[1] = 2; h_y[1] = UINT_MAX; h_z[1] = 10ULL;
    h_x[2] = 0; h_y[2] = 2342345; h_z[2] = 53483747345345ULL;
    h_x[3] = 1324423423; h_y[3] = 1; h_z[3] = 0ULL;
    h_y[4] = 575675676;
    h_x[5] = 12;

    HIP_CHECK(hipMemcpy(x, h_x, size * sizeof(unsigned int), hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(y, h_y, size * sizeof(unsigned int), hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(z, h_z, size * sizeof(unsigned long long), hipMemcpyDefault));

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(mad_u64_u32_kernel),
        dim3(1), dim3(1), 0, 0,
        x, y, z, r
    );
    HIP_CHECK(hipPeekAtLastError());

    unsigned long long h_r[size];
    HIP_CHECK(hipMemcpy(h_r, r, size * sizeof(unsigned long long), hipMemcpyDefault));

    EXPECT_EQ(h_r[0], mad_u64_u32(h_x[0], h_y[0], h_z[0]));
    EXPECT_EQ(h_r[1], mad_u64_u32(h_x[1], h_y[1], h_z[1]));
    EXPECT_EQ(h_r[2], mad_u64_u32(h_x[2], h_y[2], h_z[2]));
    EXPECT_EQ(h_r[3], mad_u64_u32(h_x[3], h_y[3], h_z[3]));
    EXPECT_EQ(h_r[4], mad_u64_u32(1403580, h_y[4], 5ULL));
    EXPECT_EQ(h_r[5], mad_u64_u32(h_x[5], 1370589, 0ULL));
    EXPECT_EQ(h_r[6], mad_u64_u32(0xFFFFFFFF, 0x87654321, 0x1234567890123456ULL));
    EXPECT_EQ(h_r[7], mad_u64_u32(23, 45, 67ULL));

    HIP_CHECK(hipFree(x));
    HIP_CHECK(hipFree(y));
    HIP_CHECK(hipFree(z));
    HIP_CHECK(hipFree(r));
}

TEST(rocrand_mrg32k3a_prng_tests, uniform_uint_test)
{
    const size_t size = 1313;
    unsigned int * data;
    HIP_CHECK(hipMalloc(&data, sizeof(unsigned int) * size));

    rocrand_mrg32k3a g;
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

TEST(rocrand_mrg32k3a_prng_tests, uniform_float_test)
{
    const size_t size = 1313;
    float * data;
    hipMalloc(&data, sizeof(float) * size);

    rocrand_mrg32k3a g;
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

TEST(rocrand_mrg32k3a_prng_tests, normal_float_test)
{
    const size_t size = 1314;
    float * data;
    hipMalloc(&data, sizeof(float) * size);

    rocrand_mrg32k3a g;
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

TEST(rocrand_mrg32k3a_prng_tests, poisson_test)
{
    const size_t size = 1313;
    unsigned int * data;
    HIP_CHECK(hipMalloc(&data, sizeof(unsigned int) * size));

    rocrand_mrg32k3a g;
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
TEST(rocrand_mrg32k3a_prng_tests, state_progress_test)
{
    // Device data
    const size_t size = 1025;
    unsigned int * data;
    HIP_CHECK(hipMalloc(&data, sizeof(unsigned int) * size));

    // Generator
    rocrand_mrg32k3a g0;

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
        if(host_data1[i] == host_data2[i]) same++;
    }
    // It may happen that numbers are the same, so we
    // just make sure that most of them are different.
    EXPECT_LT(same, static_cast<size_t>(0.01f * size));

    HIP_CHECK(hipFree(data));
}

// Checks if generators with the same seed and in the same state
// generate the same numbers
TEST(rocrand_mrg32k3a_prng_tests, same_seed_test)
{
    const unsigned long long seed = 0xdeadbeefdeadbeefULL;

    // Device side data
    const size_t size = 1024;
    unsigned int * data;
    HIP_CHECK(hipMalloc(&data, sizeof(unsigned int) * size));

    // Generators
    rocrand_mrg32k3a g0, g1;
    // Set same seeds
    g0.set_seed(seed);
    g1.set_seed(seed);

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
    // seed should be the same
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(g0_host_data[i], g1_host_data[i]);
    }

    HIP_CHECK(hipFree(data));
}

// Checks if generators with the same seed and in the same state generate
// the same numbers
TEST(rocrand_mrg32k3a_prng_tests, different_seed_test)
{
    const unsigned long long seed0 = 0xdeadbeefdeadbeefULL;
    const unsigned long long seed1 = 0xbeefdeadbeefdeadULL;

    // Device side data
    const size_t size = 1024;
    unsigned int * data;
    HIP_CHECK(hipMalloc(&data, sizeof(unsigned int) * size));

    // Generators
    rocrand_mrg32k3a g0, g1;
    // Set different seeds
    g0.set_seed(seed0);
    g1.set_seed(seed1);
    ASSERT_NE(g0.get_seed(), g1.get_seed());

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
        if(g1_host_data[i] == g0_host_data[i]) same++;
    }
    // It may happen that numbers are the same, so we
    // just make sure that most of them are different.
    EXPECT_LT(same, static_cast<size_t>(0.01f * size));

    HIP_CHECK(hipFree(data));
}

TEST(rocrand_mrg32k3a_prng_tests, discard_test)
{
    const unsigned long long seed = 12345ULL;
    rocrand_mrg32k3a::engine_type engine1(seed, 0, 678ULL);
    rocrand_mrg32k3a::engine_type engine2(seed, 0, 677ULL);

    (void)engine2.next();

    EXPECT_EQ(engine1(), engine2());

    const unsigned long long ds[] = {
        1ULL, 4ULL, 37ULL, 583ULL, 7452ULL,
        21032ULL, 35678ULL, 66778ULL, 10313475ULL, 82120230ULL
    };

    for (auto d : ds)
    {
        for (unsigned long long i = 0; i < d; i++)
        {
            (void)engine1.next();
        }
        engine2.discard(d);

        EXPECT_EQ(engine1(), engine2());
    }
}

TEST(rocrand_mrg32k3a_prng_tests, discard_sequence_test)
{
    const unsigned long long seed = 23456ULL;
    rocrand_mrg32k3a::engine_type engine1(seed, 123ULL, 444ULL);
    rocrand_mrg32k3a::engine_type engine2(seed, 123ULL, 444ULL);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard( 5356446450ULL);
    engine1.discard_sequence(123ULL);
    engine1.discard(30000000006ULL);

    engine2.discard_sequence(3ULL);
    engine2.discard(35356446456ULL);
    engine2.discard_sequence(120ULL);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard_sequence(3456000ULL);
    engine1.discard_sequence(1000005ULL);

    engine2.discard_sequence(4456005ULL);

    EXPECT_EQ(engine1(), engine2());
}

TEST(rocrand_mrg32k3a_prng_tests, discard_subsequence_test)
{
    const unsigned long long seed = 23456ULL;
    rocrand_mrg32k3a::engine_type engine1(seed, 0, 444ULL);
    rocrand_mrg32k3a::engine_type engine2(seed, 123ULL, 444ULL);

    engine1.discard_subsequence(123ULL);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard( 5356446450ULL);
    engine1.discard_subsequence(123ULL);
    engine1.discard(30000000006ULL);

    engine2.discard_subsequence(3ULL);
    engine2.discard(35356446456ULL);
    engine2.discard_subsequence(120ULL);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard_subsequence(3456000ULL);
    engine1.discard_subsequence(1000005ULL);

    engine2.discard_subsequence(4456005ULL);

    EXPECT_EQ(engine1(), engine2());
}
