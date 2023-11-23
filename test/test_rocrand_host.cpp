// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocrand/rocrand.h>

#include <gtest/gtest.h>

#include <iomanip>
#include <vector>

#define ROCRAND_CHECK(state) ASSERT_EQ(state, ROCRAND_STATUS_SUCCESS)

constexpr const unsigned long long seeds[]            = {0, 0xAAAAAAAAAAAULL};
constexpr const size_t             seeds_count        = sizeof(seeds) / sizeof(seeds[0]);
constexpr const size_t             random_seeds_count = 2;

constexpr rocrand_rng_type host_rng_types[] = {
    ROCRAND_RNG_PSEUDO_PHILOX4_32_10,
    ROCRAND_RNG_PSEUDO_MRG31K3P,
    ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32,
    ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64,
    ROCRAND_RNG_QUASI_SOBOL32,
    ROCRAND_RNG_QUASI_SOBOL64,
};

class rocrand_generate_host_test : public ::testing::TestWithParam<rocrand_rng_type>
{};

TEST_P(rocrand_generate_host_test, int_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator_host(&generator, rng_type));

    std::vector<unsigned int> results(11111);
    for(size_t i = 0; i < seeds_count + random_seeds_count; ++i)
    {
        const auto seed = i < seeds_count ? seeds[i] : rand();
        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ROCRAND_CHECK(rocrand_set_seed(generator, seed));

        ROCRAND_CHECK(rocrand_generate(generator, results.data(), results.size()));
        // We need this because the generator is async and there is no memcpy
        // that implicitly synchronizes here
        HIP_CHECK(hipDeviceSynchronize());

        double mean = 0.0;
        for(unsigned int v : results)
        {
            mean += v;
        }

        mean /= static_cast<double>(std::numeric_limits<unsigned int>::max());
        mean /= results.size();

        ASSERT_NEAR(mean, 0.5, 0.05);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

template<typename Type, typename F>
void test_int_parity(rocrand_rng_type rng_type, F generate)
{
    rocrand_generator device_generator, host_generator;
    ROCRAND_CHECK(rocrand_create_generator(&device_generator, rng_type));
    ROCRAND_CHECK(rocrand_create_generator_host(&host_generator, rng_type));

    std::vector<Type> host_results(218192);
    std::vector<Type> device_results(host_results.size());

    Type* output;
    HIP_CHECK(hipMallocHelper(&output, host_results.size() * sizeof(Type)));

    for(size_t i = 0; i < seeds_count + random_seeds_count; ++i)
    {
        const auto seed = i < seeds_count ? seeds[i] : rand();
        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ROCRAND_CHECK(rocrand_set_seed(host_generator, seed));
        ROCRAND_CHECK(rocrand_set_seed(device_generator, seed));

        ROCRAND_CHECK(generate(host_generator, host_results.data(), host_results.size()));
        ROCRAND_CHECK(generate(device_generator, output, host_results.size()));

        HIP_CHECK(hipMemcpy(device_results.data(),
                            output,
                            host_results.size() * sizeof(Type),
                            hipMemcpyDeviceToHost));

        assert_eq(host_results, device_results);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(host_generator));
    ROCRAND_CHECK(rocrand_destroy_generator(device_generator));
    HIP_CHECK(hipFree(output));
}

TEST_P(rocrand_generate_host_test, char_parity_test)
{
    test_int_parity<unsigned char>(GetParam(), rocrand_generate_char);
}

TEST_P(rocrand_generate_host_test, short_parity_test)
{
    test_int_parity<unsigned short>(GetParam(), rocrand_generate_short);
}

TEST_P(rocrand_generate_host_test, int_parity_test)
{
    test_int_parity<unsigned int>(GetParam(), rocrand_generate);
}

template<typename Type, typename F>
void test_uniform_parity(rocrand_rng_type rng_type, F generate)
{
    rocrand_generator device_generator, host_generator;
    ROCRAND_CHECK(rocrand_create_generator(&device_generator, rng_type));
    ROCRAND_CHECK(rocrand_create_generator_host(&host_generator, rng_type));

    std::vector<Type> host_results(218192);
    std::vector<Type> device_results(host_results.size());

    Type* output;
    HIP_CHECK(hipMallocHelper(&output, host_results.size() * sizeof(Type)));

    for(size_t i = 0; i < seeds_count + random_seeds_count; ++i)
    {
        const auto seed = i < seeds_count ? seeds[i] : rand();
        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ROCRAND_CHECK(rocrand_set_seed(host_generator, seed));
        ROCRAND_CHECK(rocrand_set_seed(device_generator, seed));

        ROCRAND_CHECK(generate(host_generator, host_results.data(), host_results.size()));
        ROCRAND_CHECK(generate(device_generator, output, host_results.size()));

        HIP_CHECK(hipMemcpy(device_results.data(),
                            output,
                            host_results.size() * sizeof(Type),
                            hipMemcpyDeviceToHost));

        assert_eq(host_results, device_results);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(host_generator));
    ROCRAND_CHECK(rocrand_destroy_generator(device_generator));
    HIP_CHECK(hipFree(output));
}

TEST_P(rocrand_generate_host_test, uniform_half_parity_test)
{
    test_uniform_parity<half>(GetParam(), rocrand_generate_uniform_half);
}

TEST_P(rocrand_generate_host_test, uniform_float_parity_test)
{
    test_uniform_parity<float>(GetParam(), rocrand_generate_uniform);
}

TEST_P(rocrand_generate_host_test, uniform_double_parity_test)
{
    test_uniform_parity<double>(GetParam(), rocrand_generate_uniform_double);
}

template<typename Type, typename F>
void test_normal_parity(rocrand_rng_type rng_type, F generate, double eps)
{
    Type mean   = static_cast<Type>(-12.0);
    Type stddev = static_cast<Type>(2.4);

    rocrand_generator device_generator, host_generator;
    ROCRAND_CHECK(rocrand_create_generator(&device_generator, rng_type));
    ROCRAND_CHECK(rocrand_create_generator_host(&host_generator, rng_type));

    std::vector<Type> host_results(218192);
    std::vector<Type> device_results(host_results.size());

    Type* output;
    HIP_CHECK(hipMallocHelper(&output, host_results.size() * sizeof(Type)));

    for(size_t i = 0; i < seeds_count + random_seeds_count; ++i)
    {
        const auto seed = i < seeds_count ? seeds[i] : rand();
        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ROCRAND_CHECK(rocrand_set_seed(host_generator, seed));
        ROCRAND_CHECK(rocrand_set_seed(device_generator, seed));

        ROCRAND_CHECK(
            generate(host_generator, host_results.data(), host_results.size(), mean, stddev));
        ROCRAND_CHECK(generate(device_generator, output, host_results.size(), mean, stddev));

        HIP_CHECK(hipMemcpy(device_results.data(),
                            output,
                            host_results.size() * sizeof(Type),
                            hipMemcpyDeviceToHost));

        // This rounding is required because the sine and cosine used in box-muller used in the normal
        // distribution is slightly different from the one used on the host.
        assert_near(host_results, device_results, eps);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(host_generator));
    ROCRAND_CHECK(rocrand_destroy_generator(device_generator));
    HIP_CHECK(hipFree(output));
}

TEST_P(rocrand_generate_host_test, normal_half_parity_test)
{
    test_normal_parity<half>(GetParam(), rocrand_generate_normal_half, 0.1);
}

TEST_P(rocrand_generate_host_test, normal_float_parity_test)
{
    test_normal_parity<float>(GetParam(), rocrand_generate_normal, 0.005);
}

TEST_P(rocrand_generate_host_test, normal_double_parity_test)
{
    test_normal_parity<double>(GetParam(), rocrand_generate_normal_double, 0.000001);
}

TEST_P(rocrand_generate_host_test, log_normal_half_parity_test)
{
    test_normal_parity<half>(GetParam(), rocrand_generate_log_normal_half, 0.05);
}

TEST_P(rocrand_generate_host_test, log_normal_float_parity_test)
{
    test_normal_parity<float>(GetParam(), rocrand_generate_log_normal, 0.0001);
}

TEST_P(rocrand_generate_host_test, log_normal_double_parity_test)
{
    test_normal_parity<double>(GetParam(), rocrand_generate_log_normal_double, 0.0000001);
}

TEST_P(rocrand_generate_host_test, poisson_parity_test)
{
    const rocrand_rng_type rng_type = GetParam();
    using Type                      = unsigned int;
    double lambda                   = 1.1;

    rocrand_generator device_generator, host_generator;
    ROCRAND_CHECK(rocrand_create_generator(&device_generator, rng_type));
    ROCRAND_CHECK(rocrand_create_generator_host(&host_generator, rng_type));

    std::vector<Type> host_results(218192);
    std::vector<Type> device_results(host_results.size());

    Type* output;
    HIP_CHECK(hipMallocHelper(&output, host_results.size() * sizeof(Type)));

    for(size_t i = 0; i < seeds_count + random_seeds_count; ++i)
    {
        const auto seed = i < seeds_count ? seeds[i] : rand();
        SCOPED_TRACE(testing::Message() << "with seed = " << seed);
        ROCRAND_CHECK(rocrand_set_seed(host_generator, seed));
        ROCRAND_CHECK(rocrand_set_seed(device_generator, seed));

        ROCRAND_CHECK(rocrand_generate_poisson(host_generator,
                                               host_results.data(),
                                               host_results.size(),
                                               lambda));
        ROCRAND_CHECK(
            rocrand_generate_poisson(device_generator, output, host_results.size(), lambda));

        HIP_CHECK(hipMemcpy(device_results.data(),
                            output,
                            host_results.size() * sizeof(Type),
                            hipMemcpyDeviceToHost));
    }

    assert_eq(host_results, device_results);

    ROCRAND_CHECK(rocrand_destroy_generator(host_generator));
    ROCRAND_CHECK(rocrand_destroy_generator(device_generator));
    HIP_CHECK(hipFree(output));
}

INSTANTIATE_TEST_SUITE_P(rocrand_generate_host_test,
                         rocrand_generate_host_test,
                         ::testing::ValuesIn(host_rng_types));
