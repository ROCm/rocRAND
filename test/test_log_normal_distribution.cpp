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

    const size_t size = 4000;
    float val[size];
    log_normal_distribution<float> u(0.2f, 0.5f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i+=2)
    {
        unsigned int input[2];
        float output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i] = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}

TEST(log_normal_distribution_tests, double_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    double val[size];
    log_normal_distribution<double> u(0.2, 0.5);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i+=2)
    {
        unsigned int input[4];
        double output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        input[2] = dis(gen);
        input[3] = dis(gen);
        u(input, output);
        val[i] = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    double expected_mean = std::exp(0.2 + 0.5 * 0.5 / 2);
    double expected_std = std::sqrt((std::exp(0.5 * 0.5) - 1.0) * std::exp(2 * 0.2 + 0.5 * 0.5));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1);
}

TEST(log_normal_distribution_tests, half_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    half val[size];
    log_normal_distribution<half> u(0.2f, 0.5f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i+=2)
    {
        unsigned int input[1];
        half output[2];
        input[0] = dis(gen);
        u(input, output);
        val[i] = output[0];
        val[i + 1] = output[1];
        mean += __half2float(output[0]) + __half2float(output[1]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}

TEST(mrg_log_normal_distribution_tests, float_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, ROCRAND_MRG32K3A_M1);

    const size_t size = 4000;
    float val[size];
    mrg_log_normal_distribution<float> u(0.2f, 0.5f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i+=2)
    {
        unsigned int input[2];
        float output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i] = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}

TEST(mrg_log_normal_distribution_tests, double_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, ROCRAND_MRG32K3A_M1);

    const size_t size = 4000;
    double val[size];
    mrg_log_normal_distribution<double> u(0.2, 0.5);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i+=2)
    {
        unsigned int input[2];
        double output[2];
        input[0] = dis(gen);
        input[1] = dis(gen);
        u(input, output);
        val[i] = output[0];
        val[i + 1] = output[1];
        mean += output[0] + output[1];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    double expected_mean = std::exp(0.2 + 0.5 * 0.5 / 2);
    double expected_std = std::sqrt((std::exp(0.5 * 0.5) - 1.0) * std::exp(2 * 0.2 + 0.5 * 0.5));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1);
}

TEST(mrg_log_normal_distribution_tests, half_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis(1, ROCRAND_MRG32K3A_M1);

    const size_t size = 4000;
    half val[size];
    mrg_log_normal_distribution<half> u(0.2f, 0.5f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i+=2)
    {
        unsigned int input[1];
        half output[2];
        input[0] = dis(gen);
        u(input, output);
        val[i] = output[0];
        val[i + 1] = output[1];
        mean += __half2float(output[0]) + __half2float(output[1]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}

TEST(sobol_log_normal_distribution_tests, float_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    float val[size];
    sobol_log_normal_distribution<float> u(0.2f, 0.5f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i+=1)
    {
        unsigned int input[1];
        float output[1];
        input[0] = dis(gen);
        output[0] = u(input[0]);
        val[i] = output[0];
        mean += output[0];
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}

TEST(sobol_log_normal_distribution_tests, double_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    double val[size];
    sobol_log_normal_distribution<double> u(0.2, 0.5);

    // Calculate mean
    double mean = 0.0;
    for(size_t i = 0; i < size; i+=1)
    {
        unsigned int input[1];
        double output[1];
        input[0] = dis(gen);
        output[0] = u(input[0]);
        val[i] = output[0];
        mean += output[0];
    }
    mean = mean / size;

    // Calculate stddev
    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(val[i] - mean, 2);
    }
    std = std::sqrt(std / size);

    double expected_mean = std::exp(0.2 + 0.5 * 0.5 / 2);
    double expected_std = std::sqrt((std::exp(0.5 * 0.5) - 1.0) * std::exp(2 * 0.2 + 0.5 * 0.5));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1);
}

TEST(sobol_log_normal_distribution_tests, half_test)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<unsigned int> dis;

    const size_t size = 4000;
    half val[size];
    sobol_log_normal_distribution<half> u(0.2f, 0.5f);

    // Calculate mean
    float mean = 0.0f;
    for(size_t i = 0; i < size; i+=1)
    {
        unsigned int input[1];
        half output[1];
        input[0] = dis(gen);
        output[0] = u(input[0]);
        val[i] = output[0];
        mean += __half2float(output[0]);
    }
    mean = mean / size;

    // Calculate stddev
    float std = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(__half2float(val[i]) - mean, 2);
    }
    std = std::sqrt(std / size);

    float expected_mean = std::exp(0.2f + 0.5f * 0.5f / 2);
    float expected_std = std::sqrt((std::exp(0.5f * 0.5f) - 1.0) * std::exp(2 * 0.2f + 0.5f * 0.5f));

    EXPECT_NEAR(expected_mean, mean, expected_mean * 0.1f);
    EXPECT_NEAR(expected_std, std, expected_std * 0.1f);
}
