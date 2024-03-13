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

#include <gtest/gtest.h>
#include <stdio.h>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include <rng/distribution/uniform.hpp>
#include <rng/generator_type.hpp>

#include <vector>

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)

struct dummy_generator : rocrand_generator_impl_base
{
    dummy_generator() : rocrand_generator_impl_base(ROCRAND_ORDERING_PSEUDO_DEFAULT, 0, 0) {}

    static constexpr rocrand_rng_type type()
    {
        return static_cast<rocrand_rng_type>(0);
    }

    void reset() override final
    {
        m_reset = true;
    }

    unsigned long long get_seed() const
    {
        return m_seed;
    }

    void set_seed(unsigned long long seed)
    {
        m_seed = seed;
    }

    rocrand_status set_order(rocrand_ordering order)
    {
        (void)order;
        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status init()
    {
        return ROCRAND_STATUS_SUCCESS;
    }

    template<typename T, typename Distribution = uniform_distribution<T, unsigned long long>>
    rocrand_status generate(T* data, size_t data_size, Distribution dis = Distribution())
    {
        (void)dis;
        memset(data, 0xAA, data_size * sizeof(*data));
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    template<typename T>
    rocrand_status generate_uniform(T* data, size_t data_size)
    {
        memset(data, 0xAA, data_size * sizeof(*data));
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    template<typename T>
    rocrand_status generate_normal(T* data, size_t data_size, T mean, T stddev)
    {
        memset(data, 0xAA, data_size * sizeof(*data));
        (void)mean;
        (void)stddev;
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    template<typename T>
    rocrand_status generate_log_normal(T* data, size_t data_size, T mean, T stddev)
    {
        memset(data, 0xAA, data_size * sizeof(*data));
        (void)mean;
        (void)stddev;
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    rocrand_status generate_poisson(unsigned int* data, size_t data_size, double lambda)
    {
        memset(data, 0xAA, data_size * sizeof(*data));
        (void)lambda;
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    bool               m_reset = false;
    unsigned long long m_seed  = 0;
};

TEST(rocrand_generator_type_tests, rocrand_generator)
{
    rocrand_generator g = NULL;
    EXPECT_EQ(g, static_cast<rocrand_generator>(0));

    g       = new rocrand_generator_type<dummy_generator>;
    auto gg = static_cast<rocrand_generator_base_type*>(g);
    EXPECT_NE(gg, static_cast<rocrand_generator>(0));
    EXPECT_EQ(gg->get_offset(), 0ULL);
    EXPECT_EQ(gg->set_offset(123), ROCRAND_STATUS_SUCCESS);
    EXPECT_EQ(gg->get_offset(), 123);
    EXPECT_EQ(gg->set_seed_uint4({0, 0, 0, 0}), ROCRAND_STATUS_TYPE_ERROR);
    EXPECT_EQ(gg->get_seed(), 0ULL);
    EXPECT_EQ(gg->get_stream(), (hipStream_t)(0));
    EXPECT_EQ(gg->set_dimensions(123), ROCRAND_STATUS_TYPE_ERROR);
    delete g;
}

TEST(rocrand_generator_type_tests, set_stream_test)
{
    rocrand_generator_type<dummy_generator> g;
    EXPECT_EQ(g.get_stream(), (hipStream_t)(0));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    g.set_stream(stream);
    EXPECT_EQ(g.get_stream(), stream);
    g.set_stream(NULL);
    EXPECT_EQ(g.get_stream(), (hipStream_t)(0));
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST(rocrand_generator_type_tests, generate_test)
{
    rocrand_generator_type<dummy_generator> g;

    std::vector<unsigned short> output(123);
    g.generate_short(output.data(), output.size());

    std::vector<unsigned short> expected(output.size(), 0xAAAA);
    EXPECT_EQ(expected, output);
}
