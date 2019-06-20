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

#include <random>

#include <rng/generators.hpp>

TEST(mrg_uniform_distribution_tests, uint_test)
{
    rocrand_host::detail::mrg_uniform_distribution<unsigned int> u;
    unsigned int input[1];
    unsigned int output[1];

    input[0] = ROCRAND_MRG32K3A_M1;
    u(input, output);
    EXPECT_EQ(output[0], UINT_MAX);
    input[0] = 1U;
    u(input, output);
    EXPECT_EQ(output[0], 0U);
}

TEST(mrg_uniform_distribution_tests, float_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, ROCRAND_MRG32K3A_M1);

    rocrand_host::detail::mrg_uniform_distribution<float> u;
    unsigned int input[1];
    float output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(output[0], 1.0f);
        EXPECT_GT(output[0], 0.0f);
    }

    input[0] = ROCRAND_MRG32K3A_M1;
    u(input, output);
    EXPECT_EQ(output[0], 1.0f);
    input[0] = 1U;
    u(input, output);
    EXPECT_GT(output[0], 0.0f);
    EXPECT_LT(output[0], 1e-9f);
}

TEST(mrg_uniform_distribution_tests, double_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, ROCRAND_MRG32K3A_M1);

    rocrand_host::detail::mrg_uniform_distribution<double> u;
    unsigned int input[1];
    double output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(output[0], 1.0);
        EXPECT_GT(output[0], 0.0);
    }

    input[0] = ROCRAND_MRG32K3A_M1;
    u(input, output);
    EXPECT_EQ(output[0], 1.0);
    input[0] = 1U;
    u(input, output);
    EXPECT_GT(output[0], 0.0);
    EXPECT_LT(output[0], 1e-9);
}

TEST(mrg_uniform_distribution_tests, half_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, ROCRAND_MRG32K3A_M1);

    rocrand_host::detail::mrg_uniform_distribution<half> u;
    unsigned int input[1];
    half output[2];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(__half2float(output[0]), 1.0f);
        EXPECT_LE(__half2float(output[1]), 1.0f);
        EXPECT_GT(__half2float(output[0]), 0.0f);
        EXPECT_GT(__half2float(output[1]), 0.0f);
    }

    input[0] = ROCRAND_MRG32K3A_M1;
    u(input, output);
    EXPECT_EQ(__half2float(output[0]), 1.0f);
    EXPECT_EQ(__half2float(output[1]), 1.0f);
    input[0] = 1U;
    u(input, output);
    EXPECT_GT(__half2float(output[0]), 0.0f);
    EXPECT_LT(__half2float(output[0]), 1e-4f);
    EXPECT_GT(__half2float(output[1]), 0.0f);
    EXPECT_LT(__half2float(output[1]), 1e-4f);
}

TEST(mtgp_uniform_distribution_tests, uint_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::mtgp_uniform_distribution<unsigned int> u;
    unsigned int input[1];
    unsigned int output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        unsigned int x = dis(gen);
        input[0] = x;
        u(input, output);
        EXPECT_EQ(output[0], x);
    }

    input[0] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(output[0], UINT_MAX);
    input[0] = 0U;
    u(input, output);
    EXPECT_EQ(output[0], 0U);
}

TEST(mtgp_uniform_distribution_tests, float_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::mtgp_uniform_distribution<float> u;
    unsigned int input[1];
    float output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(output[0], 1.0f);
        EXPECT_GT(output[0], 0.0f);
    }

    input[0] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(output[0], 1.0f);
    input[0] = 0U;
    u(input, output);
    EXPECT_GT(output[0], 0.0f);
    EXPECT_LT(output[0], 1e-9f);
}

TEST(mtgp_uniform_distribution_tests, double_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::mtgp_uniform_distribution<double> u;
    unsigned int input[2];
    double output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        EXPECT_LE(output[0], 1.0);
        EXPECT_GT(output[0], 0.0);
    }

    input[0] = UINT_MAX;
    input[1] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(output[0], 1.0);
    input[0] = 0U;
    input[1] = 0U;
    u(input, output);
    EXPECT_GT(output[0], 0.0);
    EXPECT_LT(output[0], 1e-9);
}

TEST(mtgp_uniform_distribution_tests, half_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::mtgp_uniform_distribution<half> u;
    unsigned int input[1];
    half output[2];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(__half2float(output[0]), 1.0f);
        EXPECT_LE(__half2float(output[1]), 1.0f);
        EXPECT_GT(__half2float(output[0]), 0.0f);
        EXPECT_GT(__half2float(output[1]), 0.0f);
    }

    input[0] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(__half2float(output[0]), 1.0f);
    EXPECT_EQ(__half2float(output[1]), 1.0f);
    input[0] = 0U;
    u(input, output);
    EXPECT_GT(__half2float(output[0]), 0.0f);
    EXPECT_LT(__half2float(output[0]), 1e-4f);
    EXPECT_GT(__half2float(output[1]), 0.0f);
    EXPECT_LT(__half2float(output[1]), 1e-4f);
}

TEST(philox_uniform_distribution_tests, uint_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::philox_uniform_distribution<unsigned int> u;
    unsigned int input[1];
    unsigned int output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        unsigned int x = dis(gen);
        input[0] = x;
        u(input, output);
        EXPECT_EQ(output[0], x);
    }

    input[0] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(output[0], UINT_MAX);
    input[0] = 0U;
    u(input, output);
    EXPECT_EQ(output[0], 0U);
}

TEST(philox_uniform_distribution_tests, float_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::philox_uniform_distribution<float> u;
    unsigned int input[1];
    float output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(output[0], 1.0f);
        EXPECT_GT(output[0], 0.0f);
    }

    input[0] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(output[0], 1.0f);
    input[0] = 0U;
    u(input, output);
    EXPECT_GT(output[0], 0.0f);
    EXPECT_LT(output[0], 1e-9f);
}

TEST(philox_uniform_distribution_tests, double_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::philox_uniform_distribution<double> u;
    unsigned int input[2];
    double output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        EXPECT_LE(output[0], 1.0);
        EXPECT_GT(output[0], 0.0);
    }

    input[0] = UINT_MAX;
    input[1] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(output[0], 1.0);
    input[0] = 0U;
    input[1] = 0U;
    u(input, output);
    EXPECT_GT(output[0], 0.0);
    EXPECT_LT(output[0], 1e-9);
}

TEST(philox_uniform_distribution_tests, half_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::philox_uniform_distribution<half> u;
    unsigned int input[1];
    half output[2];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        u(input, output);
        EXPECT_LE(__half2float(output[0]), 1.0f);
        EXPECT_LE(__half2float(output[1]), 1.0f);
        EXPECT_GT(__half2float(output[0]), 0.0f);
        EXPECT_GT(__half2float(output[1]), 0.0f);
    }

    input[0] = UINT_MAX;
    u(input, output);
    EXPECT_EQ(__half2float(output[0]), 1.0f);
    EXPECT_EQ(__half2float(output[1]), 1.0f);
    input[0] = 0U;
    u(input, output);
    EXPECT_GT(__half2float(output[0]), 0.0f);
    EXPECT_LT(__half2float(output[0]), 1e-4f);
    EXPECT_GT(__half2float(output[1]), 0.0f);
    EXPECT_LT(__half2float(output[1]), 1e-4f);
}

TEST(sobol_uniform_distribution_tests, uint_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::sobol_uniform_distribution<unsigned int> u;
    unsigned int input[1];
    unsigned int output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        unsigned int x = dis(gen);
        input[0] = x;
        output[0] = u(input[0]);
        EXPECT_EQ(output[0], x);
    }

    input[0] = UINT_MAX;
    output[0] = u(input[0]);
    EXPECT_EQ(output[0], UINT_MAX);
    input[0] = 0U;
    output[0] = u(input[0]);
    EXPECT_EQ(output[0], 0U);
}

TEST(sobol_uniform_distribution_tests, float_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::sobol_uniform_distribution<float> u;
    unsigned int input[1];
    float output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        output[0] = u(input[0]);
        EXPECT_LE(output[0], 1.0f);
        EXPECT_GT(output[0], 0.0f);
    }

    input[0] = UINT_MAX;
    output[0] = u(input[0]);
    EXPECT_EQ(output[0], 1.0f);
    input[0] = 0U;
    output[0] = u(input[0]);
    EXPECT_GT(output[0], 0.0f);
    EXPECT_LT(output[0], 1e-9f);
}

TEST(sobol_uniform_distribution_tests, double_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::sobol_uniform_distribution<double> u;
    unsigned int input[1];
    double output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        output[0] = u(input[0]);
        EXPECT_LE(output[0], 1.0);
        EXPECT_GT(output[0], 0.0);
    }

    input[0] = UINT_MAX;
    output[0] = u(input[0]);
    EXPECT_EQ(output[0], 1.0);
    input[0] = 0U;
    output[0] = u(input[0]);
    EXPECT_GT(output[0], 0.0);
    EXPECT_LT(output[0], 1e-9);
}

TEST(sobol_uniform_distribution_tests, half_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    rocrand_host::detail::sobol_uniform_distribution<half> u;
    unsigned int input[1];
    half output[1];
    for(size_t i = 0; i < 1000000; i++)
    {
        input[0] = dis(gen);
        output[0] = u(input[0]);
        EXPECT_LE(__half2float(output[0]), 1.0f);
        EXPECT_GT(__half2float(output[0]), 0.0f);
    }

    input[0] = UINT_MAX;
    output[0] = u(input[0]);
    EXPECT_EQ(__half2float(output[0]), 1.0f);
    input[0] = 0U;
    output[0] = u(input[0]);
    EXPECT_GT(__half2float(output[0]), 0.0f);
    EXPECT_LT(__half2float(output[0]), 1e-4f);
}
