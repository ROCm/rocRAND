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
#include "test_rocrand_prng.hpp"
#include <rocrand/rocrand.h>

#include <rng/mrg.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

// Generator API tests
using rocrand_mrg_generator_prng_tests_types = ::testing::Types<
    generator_prng_tests_params<rocrand_mrg31k3p, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<rocrand_mrg31k3p, ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    generator_prng_tests_params<rocrand_mrg32k3a, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_tests_params<rocrand_mrg32k3a, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

using rocrand_mrg_generator_prng_offset_tests_types = ::testing::Types<
    generator_prng_offset_tests_params<unsigned int,
                                       rocrand_mrg31k3p,
                                       ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<unsigned int,
                                       rocrand_mrg31k3p,
                                       ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    generator_prng_offset_tests_params<unsigned int,
                                       rocrand_mrg32k3a,
                                       ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<unsigned int,
                                       rocrand_mrg32k3a,
                                       ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    generator_prng_offset_tests_params<float, rocrand_mrg31k3p, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<float, rocrand_mrg31k3p, ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
    generator_prng_offset_tests_params<float, rocrand_mrg32k3a, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    generator_prng_offset_tests_params<float, rocrand_mrg32k3a, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_mrg,
                               generator_prng_tests,
                               rocrand_mrg_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_mrg,
                               generator_prng_continuity_tests,
                               rocrand_mrg_generator_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_mrg,
                               generator_prng_offset_tests,
                               rocrand_mrg_generator_prng_offset_tests_types);

// mrg-specific generator API tests
template<class Params>
struct rocrand_mrg_generator_prng_tests : public ::testing::Test
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

TYPED_TEST_SUITE(rocrand_mrg_generator_prng_tests, rocrand_mrg_generator_prng_tests_types);

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

TEST(rocrand_mrg_generator_prng_tests, mad_u64_u32_test)
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

template<class Generator, class T>
void uniform_floating_point_range_test(rocrand_ordering ordering)
{
    const size_t size = 1 << 26;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    Generator g;
    g.set_order(ordering);
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

TYPED_TEST(rocrand_mrg_generator_prng_tests, uniform_float_range_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;

    uniform_floating_point_range_test<generator_t, float>(ordering);
}

TYPED_TEST(rocrand_mrg_generator_prng_tests, uniform_double_range_test)
{
    using generator_t                   = typename TestFixture::generator_t;
    constexpr rocrand_ordering ordering = TestFixture::ordering;

    uniform_floating_point_range_test<generator_t, double>(ordering);
}

TYPED_TEST(rocrand_mrg_generator_prng_tests, different_seed_test)
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
