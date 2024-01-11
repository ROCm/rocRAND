// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_TEST_INTERNAL_TEST_ROCRAND_THREEFRY_PRNG_HPP_
#define ROCRAND_TEST_INTERNAL_TEST_ROCRAND_THREEFRY_PRNG_HPP_

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include <rocrand/rocrand.h>

#include <rng/threefry.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

// threefryNx64_20-specific generator API tests
template<class Params>
struct threefryNx64_20_generator_prng_tests : public ::testing::Test
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

TYPED_TEST_SUITE_P(threefryNx64_20_generator_prng_tests);

// Assert that the kernel arguments are trivially copyable and destructible.
TYPED_TEST_P(threefryNx64_20_generator_prng_tests, type)
{
    using generator_t = typename TestFixture::generator_t;
    using engine_type = typename generator_t::engine_type::base_type;
    // TODO: Enable once uint2 is trivially copyable.
    // EXPECT_TRUE(std::is_trivially_copyable<engine_type>::value);
    EXPECT_TRUE(std::is_trivially_destructible<engine_type>::value);
}

TYPED_TEST_P(threefryNx64_20_generator_prng_tests, uniform_ulonglong_test)
{
    const size_t        size = 1313;
    unsigned long long* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned long long) * (size + 1)));

    auto g = TestFixture::get_generator();
    ROCRAND_CHECK(g.generate_uniform(data + 1, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long host_data[size];
    HIP_CHECK(
        hipMemcpy(host_data, data + 1, sizeof(unsigned long long) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i] / static_cast<double>(size);
    }
    ASSERT_NEAR(mean, static_cast<double>(ULLONG_MAX / 2), static_cast<double>(ULLONG_MAX / 20));

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(threefryNx64_20_generator_prng_tests, different_seed_test)
{
    const unsigned long long seed0 = 0xdeadbeefdeadbeefULL;
    const unsigned long long seed1 = 0xbeefdeadbeefdeadULL;

    // Device side data
    const size_t        size = 1024;
    unsigned long long* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned long long) * size));

    // Generators
    auto g0 = TestFixture::get_generator(), g1 = TestFixture::get_generator();

    // Set different seeds
    g0.set_seed(seed0);
    g1.set_seed(seed1);
    ASSERT_NE(g0.get_seed(), g1.get_seed());

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long g0_host_data[size];
    HIP_CHECK(
        hipMemcpy(g0_host_data, data, sizeof(unsigned long long) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Generate using g1 and copy to host
    ROCRAND_CHECK(g1.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long g1_host_data[size];
    HIP_CHECK(
        hipMemcpy(g1_host_data, data, sizeof(unsigned long long) * size, hipMemcpyDeviceToHost));
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

REGISTER_TYPED_TEST_SUITE_P(threefryNx64_20_generator_prng_tests,
                            type,
                            uniform_ulonglong_test,
                            different_seed_test);

// threefryNx64_20-specific generator API continuity tests
template<class Params>
struct threefryNx64_20_generator_prng_continuity_tests : public ::testing::Test
{
    using generator_t                                 = typename Params::generator_t;
    static inline constexpr rocrand_ordering ordering = Params::ordering;
};

TYPED_TEST_SUITE_P(threefryNx64_20_generator_prng_continuity_tests);

TYPED_TEST_P(threefryNx64_20_generator_prng_continuity_tests, continuity_uniform_ullong_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef unsigned long long int output_t;

    continuity_test<output_t, generator_t>(
        [](generator_t& g, output_t* data, size_t s) { g.generate_uniform(data, s); },
        ordering,
        uniform_distribution<output_t, unsigned long long int>::output_width);
}

REGISTER_TYPED_TEST_SUITE_P(threefryNx64_20_generator_prng_continuity_tests,
                            continuity_uniform_ullong_test);

#endif // ROCRAND_TEST_INTERNAL_TEST_ROCRAND_THREEFRY_PRNG_HPP_
