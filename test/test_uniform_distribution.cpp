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

#include <rng/distribution/uniform.hpp>

TEST(uniform_distribution_tests, uint_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    uniform_distribution<unsigned int> u;
    for(size_t i = 0; i < 100; i++)
    {
        unsigned int x = dis(gen);
        EXPECT_EQ(u(x), x);
    }

    EXPECT_EQ(u(UINT_MAX), UINT_MAX);
    EXPECT_EQ(u(0), 0U);
}

TEST(uniform_distribution_tests, float_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    uniform_distribution<float> u;
    for(size_t i = 0; i < 100; i++)
    {
        unsigned int x = dis(gen);
        EXPECT_LE(u(x), 1.0f);
        EXPECT_GT(u(x), 0.0f);
    }

    EXPECT_EQ(u(UINT_MAX), 1.0f);
    EXPECT_GT(u(0), 0.0f);
}

TEST(uniform_distribution_tests, double_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    uniform_distribution<double> u;
    for(size_t i = 0; i < 100; i++)
    {
        unsigned int x = dis(gen);
        EXPECT_LE(u(x), 1.0);
        EXPECT_GT(u(x), 0.0);
    }

    EXPECT_EQ(u(ULLONG_MAX), 1.0);
    EXPECT_EQ(u(UINT_MAX), 1.0);
    EXPECT_GT(u(0U), 0.0);
}
