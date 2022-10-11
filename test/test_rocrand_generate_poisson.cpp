// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocrand/rocrand.h>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"

class rocrand_generate_poisson_tests : public ::testing::TestWithParam<rocrand_rng_type>
{};

template<typename T, typename GenerateFunc>
void test_generate(GenerateFunc generate_func)
{
    const rocrand_rng_type rng_type = rocrand_generate_poisson_tests::GetParam();

    if(sizeof(T) == 4
       && (rng_type == ROCRAND_RNG_PSEUDO_THREEFRY2_64_20
           || rng_type == ROCRAND_RNG_PSEUDO_THREEFRY4_64_20))
    {
        GTEST_SKIP() << "rocrand_generate_poisson not implemented for 64-bits Threefry";
    }

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

    const size_t size   = 12563;
    double       lambda = 100.0;
    T*           data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), size * sizeof(T)));
    HIP_CHECK(hipDeviceSynchronize());

    ROCRAND_CHECK(generate_func(generator, data, size, lambda));

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

template<typename T, typename GenerateFunc>
void test_out_of_range(GenerateFunc generate_func)
{
    const rocrand_rng_type rng_type = rocrand_generate_poisson_tests::GetParam();

    if(sizeof(T) == 4
       && (rng_type == ROCRAND_RNG_PSEUDO_THREEFRY2_64_20
           || rng_type == ROCRAND_RNG_PSEUDO_THREEFRY4_64_20))
    {
        GTEST_SKIP() << "rocrand_generate_poisson not implemented for 64-bits Threefry";
    }

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

    const size_t size   = 256;
    double       lambda = 0.0;
    T*           data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), size * sizeof(T)));
    HIP_CHECK(hipDeviceSynchronize());

    EXPECT_EQ(generate_func(generator, data, size, lambda), ROCRAND_STATUS_OUT_OF_RANGE);

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST_P(rocrand_generate_poisson_tests, generate_test)
{
    test_generate<unsigned int>(
        [](rocrand_generator gen, unsigned int* data, size_t size, double lambda)
        { return rocrand_generate_poisson(gen, data, size, lambda); });
}

TEST(rocrand_generate_poisson_tests, neg_test)
{
    const size_t size = 256;
    double         lambda = 100.0;
    unsigned int * data = NULL;

    rocrand_generator generator = NULL;
    EXPECT_EQ(
        rocrand_generate_poisson(generator, (unsigned int *)data, size, lambda),
        ROCRAND_STATUS_NOT_CREATED
    );
}

TEST_P(rocrand_generate_poisson_tests, out_of_range_test)
{
    test_out_of_range<unsigned int>(
        [](rocrand_generator gen, unsigned int* data, size_t size, double lambda)
        { return rocrand_generate_poisson(gen, data, size, lambda); });
}

INSTANTIATE_TEST_SUITE_P(rocrand_generate_poisson_tests,
                         rocrand_generate_poisson_tests,
                         ::testing::ValuesIn(rng_types));

class rocrand_generate_poisson_64_tests : public ::testing::TestWithParam<rocrand_rng_type>
{};

TEST_P(rocrand_generate_poisson_64_tests, generate_test)
{
    test_generate<unsigned long long int>(
        [](rocrand_generator gen, unsigned long long int* data, size_t size, double lambda)
        { return rocrand_generate_poisson_long_long(gen, data, size, lambda); });
}

TEST(rocrand_generate_poisson_64_tests, neg_test)
{
    const size_t        size   = 256;
    double              lambda = 100.0;
    unsigned long long* data   = NULL;

    rocrand_generator generator = NULL;
    EXPECT_EQ(
        rocrand_generate_poisson_long_long(generator, (unsigned long long*)data, size, lambda),
        ROCRAND_STATUS_NOT_CREATED);
}

TEST_P(rocrand_generate_poisson_64_tests, out_of_range_test)
{
    test_out_of_range<unsigned long long int>(
        [](rocrand_generator gen, unsigned long long int* data, size_t size, double lambda)
        { return rocrand_generate_poisson_long_long(gen, data, size, lambda); });
}

INSTANTIATE_TEST_SUITE_P(rocrand_generate_poisson_64_tests,
                         rocrand_generate_poisson_64_tests,
                         ::testing::ValuesIn(long_long_rng_types));
