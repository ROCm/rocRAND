// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include <rocrand/rocrand.h>

#include <rng/mrg.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <stdio.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

template<class Params>
struct rocrand_mrg_prng_tests : public ::testing::Test
{
    using generator_t                                 = typename Params::generator_t;
    static inline constexpr rocrand_ordering ordering = Params::ordering;

    auto get_generator() const
    {
        generator_t g;
        if(g.set_order(ordering) != ROCRAND_STATUS_SUCCESS)
        {
            throw std::runtime_error("Could not set ordering for generator");
        }
        return g;
    }
};

template<class Generator, rocrand_ordering Ordering>
struct rocrand_mrg_prng_tests_params
{
    using generator_t                                 = Generator;
    static inline constexpr rocrand_ordering ordering = Ordering;
};

TYPED_TEST_SUITE_P(rocrand_mrg_prng_tests);

using rocrand_device::detail::mad_u64_u32;

__global__ __launch_bounds__(1) void mad_u64_u32_kernel(const unsigned int*       x,
                                                        const unsigned int*       y,
                                                        const unsigned long long* z,
                                                        unsigned long long*       r)
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

TYPED_TEST_P(rocrand_mrg_prng_tests, mad_u64_u32_test)
{
    const size_t size = 8;

    unsigned int*       x;
    unsigned int*       y;
    unsigned long long* z;
    unsigned long long* r;
    HIP_CHECK(hipMallocHelper(&x, size * sizeof(unsigned int)));
    HIP_CHECK(hipMallocHelper(&y, size * sizeof(unsigned int)));
    HIP_CHECK(hipMallocHelper(&z, size * sizeof(unsigned long long)));
    HIP_CHECK(hipMallocHelper(&r, size * sizeof(unsigned long long)));

    unsigned int       h_x[size];
    unsigned int       h_y[size];
    unsigned long long h_z[size];

    h_x[0] = 3492343451;
    h_y[0] = 1234;
    h_z[0] = 1231314234234265ULL;
    h_x[1] = 2;
    h_y[1] = UINT_MAX;
    h_z[1] = 10ULL;
    h_x[2] = 0;
    h_y[2] = 2342345;
    h_z[2] = 53483747345345ULL;
    h_x[3] = 1324423423;
    h_y[3] = 1;
    h_z[3] = 0ULL;
    h_y[4] = 575675676;
    h_x[5] = 12;

    HIP_CHECK(hipMemcpy(x, h_x, size * sizeof(unsigned int), hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(y, h_y, size * sizeof(unsigned int), hipMemcpyDefault));
    HIP_CHECK(hipMemcpy(z, h_z, size * sizeof(unsigned long long), hipMemcpyDefault));

    hipLaunchKernelGGL(HIP_KERNEL_NAME(mad_u64_u32_kernel), dim3(1), dim3(1), 0, 0, x, y, z, r);
    HIP_CHECK(hipGetLastError());

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

TYPED_TEST_P(rocrand_mrg_prng_tests, init_test)
{
    auto g = TestFixture::get_generator(); // offset = 0
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1337);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1048576);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1 << 24);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1 << 28);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset((1ULL << 36) + 1234567ULL);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());
}

TYPED_TEST_P(rocrand_mrg_prng_tests, uniform_uint_test)
{
    const size_t  size = 1313;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    auto g = TestFixture::get_generator();
    ROCRAND_CHECK(g.generate_uniform(data, size));
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

template<class Generator, class T>
void uniform_floating_point_test()
{
    const size_t size = 1313;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    Generator g;
    ROCRAND_CHECK(g.generate_uniform(data, size));

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(*host_data) * size, hipMemcpyDeviceToHost));

    double sum = 0;
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_GT(host_data[i], static_cast<T>(0.0));
        ASSERT_LE(host_data[i], static_cast<T>(1.0));
        sum += host_data[i];
    }
    const double mean = sum / size;
    ASSERT_NEAR(mean, 0.5, 0.05);

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(rocrand_mrg_prng_tests, uniform_float_test)
{
    using generator_t = typename TestFixture::generator_t;

    uniform_floating_point_test<generator_t, float>();
}

TYPED_TEST_P(rocrand_mrg_prng_tests, uniform_double_test)
{
    using generator_t = typename TestFixture::generator_t;

    uniform_floating_point_test<generator_t, double>();
}

template<class Generator, class T>
void uniform_floating_point_range_test()
{
    const size_t size = 1 << 26;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    Generator g;
    ROCRAND_CHECK(g.generate_uniform(data, size));

    T* host_data = new T[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(*host_data) * size, hipMemcpyDeviceToHost));

    for(size_t i = 0; i < size; i++)
    {
        ASSERT_GT(host_data[i], 0.0f);
        ASSERT_LE(host_data[i], 1.0f);
    }

    HIP_CHECK(hipFree(data));
    delete[] host_data;
}

TYPED_TEST_P(rocrand_mrg_prng_tests, uniform_float_range_test)
{
    using generator_t = typename TestFixture::generator_t;

    uniform_floating_point_range_test<generator_t, float>();
}

TYPED_TEST_P(rocrand_mrg_prng_tests, uniform_double_range_test)
{
    using generator_t = typename TestFixture::generator_t;

    uniform_floating_point_range_test<generator_t, double>();
}

template<class Generator, class T>
void normal_floating_point_test()
{
    const size_t size = 1313;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    Generator g;
    ROCRAND_CHECK(g.generate_normal(data, size, static_cast<T>(2.0), static_cast<T>(5.0)));
    HIP_CHECK(hipDeviceSynchronize());

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(*host_data) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;

    double stddev = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        stddev += std::pow(host_data[i] - mean, 2);
    }
    stddev = std::sqrt(stddev / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, stddev, 1.0); // 20%

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(rocrand_mrg_prng_tests, normal_float_test)
{
    using generator_t = typename TestFixture::generator_t;

    normal_floating_point_test<generator_t, float>();
}

TYPED_TEST_P(rocrand_mrg_prng_tests, normal_double_test)
{
    using generator_t = typename TestFixture::generator_t;

    normal_floating_point_test<generator_t, double>();
}

template<class Generator, class T>
void log_normal_floating_point_test()
{
    const size_t size = 131313;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    T normal_mean   = static_cast<T>(3.0);
    T normal_stddev = static_cast<T>(1.5);
    T normal_var    = normal_stddev * normal_stddev;

    T log_normal_mean   = std::exp(normal_mean + normal_var / 2.0);
    T log_normal_stddev = std::sqrt(std::exp(normal_var) - 1.0) * log_normal_mean;

    Generator g;
    ROCRAND_CHECK(g.generate_log_normal(data, size, normal_mean, normal_stddev));
    HIP_CHECK(hipDeviceSynchronize());

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(*host_data) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;

    double stddev = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        stddev += std::pow(host_data[i] - mean, 2);
    }
    stddev = std::sqrt(stddev / size);

    EXPECT_NEAR(log_normal_mean, mean, log_normal_mean * 0.2); // 20%
    EXPECT_NEAR(log_normal_stddev, stddev, log_normal_stddev * 0.2); // 20%

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(rocrand_mrg_prng_tests, log_normal_float_test)
{
    using generator_t = typename TestFixture::generator_t;

    log_normal_floating_point_test<generator_t, float>();
}

TYPED_TEST_P(rocrand_mrg_prng_tests, log_normal_double_test)
{
    using generator_t = typename TestFixture::generator_t;

    log_normal_floating_point_test<generator_t, double>();
}

TYPED_TEST_P(rocrand_mrg_prng_tests, poisson_test)
{
    const size_t  size = 1313;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    auto g = TestFixture::get_generator();
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
TYPED_TEST_P(rocrand_mrg_prng_tests, state_progress_test)
{
    // Device data
    const size_t  size = 1025;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    // Generator
    auto g0 = TestFixture::get_generator();

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
TYPED_TEST_P(rocrand_mrg_prng_tests, same_seed_test)
{
    const unsigned long long seed = 0xdeadbeefdeadbeefULL;

    // Device side data
    const size_t  size = 1024;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    // Generators
    auto g0 = TestFixture::get_generator(), g1 = TestFixture::get_generator();
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

TYPED_TEST_P(rocrand_mrg_prng_tests, different_seed_test)
{
    const unsigned long long seed0 = 0xdeadbeefdeadbeefULL;
    const unsigned long long seed1 = 0xbeefdeadbeefdeadULL;

    // Device side data
    const size_t  size = 1024;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    // Generators
    auto g0 = TestFixture::get_generator(), g1 = TestFixture::get_generator();
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
        if(g1_host_data[i] == g0_host_data[i])
            same++;
    }
    // It may happen that numbers are the same, so we
    // just make sure that most of them are different.
    EXPECT_LT(same, static_cast<size_t>(0.01f * size));

    HIP_CHECK(hipFree(data));
}

// Check that subsequent generations of different sizes produce one
// sequence without gaps, no matter how many values are generated per call.
template<typename T, typename Generator, typename GenerateFunc>
void continuity_test(GenerateFunc     generate_func,
                     rocrand_ordering ordering,
                     unsigned int     divisor = 1)
{
    std::vector<size_t> sizes0({100, 1, 24783, 3, 2, 776543});
    std::vector<size_t> sizes1({1024, 55, 65536, 623456, 30, 111331});
    if(divisor > 1)
    {
        for(size_t& s : sizes0)
            s = (s + divisor - 1) & ~static_cast<size_t>(divisor - 1);
        for(size_t& s : sizes1)
            s = (s + divisor - 1) & ~static_cast<size_t>(divisor - 1);
    }

    const size_t size0 = std::accumulate(sizes0.cbegin(), sizes0.cend(), std::size_t{0});
    const size_t size1 = std::accumulate(sizes1.cbegin(), sizes1.cend(), std::size_t{0});

    T* data0;
    T* data1;
    HIP_CHECK(hipMalloc(&data0, sizeof(T) * size0));
    HIP_CHECK(hipMalloc(&data1, sizeof(T) * size1));

    Generator g0;
    g0.set_order(ordering);
    Generator g1;
    g1.set_order(ordering);

    std::vector<T> host_data0(size0);
    std::vector<T> host_data1(size1);

    size_t current0 = 0;
    for(size_t s : sizes0)
    {
        generate_func(g0, data0, s);
        HIP_CHECK(hipMemcpy(host_data0.data() + current0, data0, sizeof(T) * s, hipMemcpyDefault));
        current0 += s;
    }
    size_t current1 = 0;
    for(size_t s : sizes1)
    {
        generate_func(g1, data1, s);
        HIP_CHECK(hipMemcpy(host_data1.data() + current1, data1, sizeof(T) * s, hipMemcpyDefault));
        current1 += s;
    }

    for(size_t i = 0; i < std::min(size0, size1); i++)
    {
        ASSERT_EQ(host_data0[i], host_data1[i]);
    }

    HIP_CHECK(hipFree(data0));
    HIP_CHECK(hipFree(data1));
}

TYPED_TEST_P(rocrand_mrg_prng_tests, continuity_uniform_uint_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef unsigned int output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_uniform(data, s); },
                                           ordering);
}

TYPED_TEST_P(rocrand_mrg_prng_tests, continuity_uniform_char_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef unsigned char output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_uniform(data, s); },
                                           ordering,
                                           4);
}

TYPED_TEST_P(rocrand_mrg_prng_tests, continuity_uniform_float_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef float output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_uniform(data, s); },
                                           ordering);
}

TYPED_TEST_P(rocrand_mrg_prng_tests, continuity_uniform_double_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef double output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_uniform(data, s); },
                                           ordering);
}

TYPED_TEST_P(rocrand_mrg_prng_tests, continuity_normal_float_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef float output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_normal(data, s, 0.0f, 1.0f); },
                                           ordering,
                                           2);
}

TYPED_TEST_P(rocrand_mrg_prng_tests, continuity_normal_double_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef double output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_normal(data, s, 0.0, 1.0); },
                                           ordering,
                                           2);
}

TYPED_TEST_P(rocrand_mrg_prng_tests, continuity_log_normal_float_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef float output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_log_normal(data, s, 0.0f, 1.0f); },
                                           ordering,
                                           2);
}

TYPED_TEST_P(rocrand_mrg_prng_tests, continuity_log_normal_double_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef double output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_log_normal(data, s, 0.0, 1.0); },
                                           ordering,
                                           2);
}

TYPED_TEST_P(rocrand_mrg_prng_tests, continuity_poisson_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef unsigned int output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_poisson(data, s, 100.0); },
                                           ordering);
}

template<class Params>
struct rocrand_mrg_prng_offset_tests : public ::testing::Test
{
    using output_t                                    = typename Params::output_t;
    using generator_t                                 = typename Params::generator_t;
    static inline constexpr rocrand_ordering ordering = Params::ordering;

    auto get_generator() const
    {
        generator_t g;
        if(g.set_order(ordering) != ROCRAND_STATUS_SUCCESS)
        {
            throw std::runtime_error("Could not set ordering for generator");
        }
        return g;
    }
};

template<class Output, class Generator, rocrand_ordering Ordering>
struct rocrand_mrg_prng_offset_tests_params
{
    using output_t                                    = Output;
    using generator_t                                 = Generator;
    static constexpr inline rocrand_ordering ordering = Ordering;
};

TYPED_TEST_SUITE_P(rocrand_mrg_prng_offset_tests);

TYPED_TEST_P(rocrand_mrg_prng_offset_tests, offsets_test)
{
    using output_t    = typename TestFixture::output_t;
    const size_t size = 131313;

    constexpr size_t offsets[] = {0, 1, 11, 112233};

    for(const auto offset : offsets)
    {
        const size_t size0 = size;
        const size_t size1 = (size + offset);
        output_t*    data0;
        output_t*    data1;
        HIP_CHECK(hipMalloc(&data0, sizeof(output_t) * size0));
        HIP_CHECK(hipMalloc(&data1, sizeof(output_t) * size1));

        auto g0 = TestFixture::get_generator();
        g0.set_offset(offset);
        g0.generate(data0, size0);

        auto g1 = TestFixture::get_generator();
        g1.generate(data1, size1);

        std::vector<output_t> host_data0(size0);
        std::vector<output_t> host_data1(size1);
        HIP_CHECK(
            hipMemcpy(host_data0.data(), data0, sizeof(output_t) * size0, hipMemcpyDeviceToHost));
        HIP_CHECK(
            hipMemcpy(host_data1.data(), data1, sizeof(output_t) * size1, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        for(size_t i = 0; i < size; ++i)
        {
            ASSERT_EQ(host_data0[i], host_data1[i + offset]);
        }

        HIP_CHECK(hipFree(data0));
        HIP_CHECK(hipFree(data1));
    }
}

REGISTER_TYPED_TEST_SUITE_P(rocrand_mrg_prng_tests,
                            mad_u64_u32_test,
                            init_test,
                            uniform_uint_test,
                            uniform_float_test,
                            uniform_double_test,
                            uniform_float_range_test,
                            uniform_double_range_test,
                            normal_float_test,
                            normal_double_test,
                            log_normal_float_test,
                            log_normal_double_test,
                            poisson_test,
                            state_progress_test,
                            same_seed_test,
                            different_seed_test,
                            continuity_uniform_uint_test,
                            continuity_uniform_char_test,
                            continuity_uniform_float_test,
                            continuity_uniform_double_test,
                            continuity_normal_float_test,
                            continuity_normal_double_test,
                            continuity_log_normal_float_test,
                            continuity_log_normal_double_test,
                            continuity_poisson_test);

REGISTER_TYPED_TEST_SUITE_P(rocrand_mrg_prng_offset_tests, offsets_test);

// Generator API tests
using rocrand_mrg_prng_tests_types = ::testing::Types<
    rocrand_mrg_prng_tests_params<rocrand_mrg31k3p, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    rocrand_mrg_prng_tests_params<rocrand_mrg31k3p, ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    rocrand_mrg_prng_tests_params<rocrand_mrg32k3a, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    rocrand_mrg_prng_tests_params<rocrand_mrg32k3a, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

using rocrand_mrg_prng_offset_tests_types = ::testing::Types<
    rocrand_mrg_prng_offset_tests_params<unsigned int,
                                         rocrand_mrg31k3p,
                                         ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    rocrand_mrg_prng_offset_tests_params<unsigned int,
                                         rocrand_mrg31k3p,
                                         ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    rocrand_mrg_prng_offset_tests_params<unsigned int,
                                         rocrand_mrg32k3a,
                                         ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    rocrand_mrg_prng_offset_tests_params<unsigned int,
                                         rocrand_mrg32k3a,
                                         ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    rocrand_mrg_prng_offset_tests_params<float, rocrand_mrg31k3p, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    rocrand_mrg_prng_offset_tests_params<float, rocrand_mrg31k3p, ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    rocrand_mrg_prng_offset_tests_params<float, rocrand_mrg32k3a, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    rocrand_mrg_prng_offset_tests_params<float, rocrand_mrg32k3a, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_mrg_prng_tests,
                               rocrand_mrg_prng_tests,
                               rocrand_mrg_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_mrg_prng_offset_tests,
                               rocrand_mrg_prng_offset_tests,
                               rocrand_mrg_prng_offset_tests_types);

// Engine API tests
template<class Generator>
struct rocrand_mrg_prng_engine_tests : public ::testing::Test
{
    using mrg_type = Generator;
};

using rocrand_mrg_prng_engine_tests_types = ::testing::Types<rocrand_mrg31k3p, rocrand_mrg32k3a>;

TYPED_TEST_SUITE(rocrand_mrg_prng_engine_tests, rocrand_mrg_prng_engine_tests_types);

TYPED_TEST(rocrand_mrg_prng_engine_tests, discard_test)
{
    typedef typename TestFixture::mrg_type mrg_type;
    const unsigned long long               seed = 12345ULL;
    typename mrg_type::engine_type         engine1(seed, 0, 678ULL);
    typename mrg_type::engine_type         engine2(seed, 0, 677ULL);

    (void)engine2.next();

    EXPECT_EQ(engine1(), engine2());

    const unsigned long long ds[] = {1ULL,
                                     4ULL,
                                     37ULL,
                                     583ULL,
                                     7452ULL,
                                     21032ULL,
                                     35678ULL,
                                     66778ULL,
                                     10313475ULL,
                                     82120230ULL};

    for(auto d : ds)
    {
        for(unsigned long long i = 0; i < d; i++)
        {
            (void)engine1.next();
        }
        engine2.discard(d);

        EXPECT_EQ(engine1(), engine2());
    }
}

TYPED_TEST(rocrand_mrg_prng_engine_tests, discard_sequence_test)
{
    typedef typename TestFixture::mrg_type mrg_type;
    const unsigned long long               seed = 23456ULL;
    typename mrg_type::engine_type         engine1(seed, 123ULL, 444ULL);
    typename mrg_type::engine_type         engine2(seed, 123ULL, 444ULL);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard(5356446450ULL);
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

TYPED_TEST(rocrand_mrg_prng_engine_tests, discard_subsequence_test)
{
    typedef typename TestFixture::mrg_type mrg_type;
    const unsigned long long               seed = 23456ULL;
    typename mrg_type::engine_type         engine1(seed, 0, 444ULL);
    typename mrg_type::engine_type         engine2(seed, 123ULL, 444ULL);

    engine1.discard_subsequence(123ULL);

    EXPECT_EQ(engine1(), engine2());

    engine1.discard(5356446450ULL);
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
