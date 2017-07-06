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

#include <rng/distribution/log_normal.hpp>

TEST(log_normal_distribution_tests, float_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    float mean = 0.0f;
    float std = 0.0f;
    const size_t size = 4000;
    const size_t half = size / 2;
    log_normal_distribution<float> u(5.0f, 2.0f);
    float val[size];
    size_t z = 0;
    for(size_t i = 0; i < half; i++)
    {
        unsigned int x = dis(gen);
        unsigned int y = dis(gen);
        float2 v = u(x, y);
        val[z] = v.x;
        val[z + 1] = v.y;
        mean += v.x + v.y;
        z += 2;
    }

    mean = mean / size;
    z = 0;

    for(size_t i = 0; i < half; i++)
    {
        std += powf(val[z] - mean, 2);
        std += powf(val[z + 1] - mean, 2);
        z += 2;
    }
    std = sqrtf(std / size);

    float logmean = logf(mean * mean / sqrtf(std * std + mean * mean));
    float logstd = sqrtf(logf(std * std / mean / mean + 1.0f));

    EXPECT_NEAR(5.0f, logmean, 1.0f);
    EXPECT_NEAR(2.0f, logstd, 1.0f);
}

TEST(log_normal_distribution_tests, double_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    double mean = 0.0;
    double std = 0.0;
    const size_t size = 4000;
    const size_t half = size / 2;
    log_normal_distribution<double> u(5.0, 2.0);
    double val[size];
    size_t z1 = 0;
    for(size_t i = 0; i < half; i++)
    {
        unsigned int x = dis(gen);
        unsigned int y = dis(gen);
        unsigned int z = dis(gen);
        unsigned int w = dis(gen);
        uint4 t = { x, y, z, w };
        double2 v = u(t);
        val[z1] = v.x;
        val[z1 + 1] = v.y;
        mean += v.x + v.y;
        z1 += 2;
    }

    mean = mean / size;
    z1 = 0;

    for(size_t i = 0; i < half; i++)
    {
        std += pow(val[z1] - mean, 2);
        std += pow(val[z1 + 1] - mean, 2);
        z1 += 2;
    }
    std = sqrt(std / size);

    double logmean = log(mean * mean / sqrt(std * std + mean * mean));
    double logstd = sqrt(log(std * std / mean / mean + 1.0));

    EXPECT_NEAR(5.0, logmean, 1.0);
    EXPECT_NEAR(2.0, logstd, 1.0);
}
