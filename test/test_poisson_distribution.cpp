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
#include <vector>

#include <rng/generator_type.hpp>
#include <rng/generators.hpp>
#include <rng/distribution/poisson.hpp>

template<typename T>
double get_mean(std::vector<T> values)
{
    double mean = 0.0f;
    for (auto v : values)
    {
        mean += static_cast<double>(v);
    }
    return mean / values.size();
}

template<typename T>
double get_variance(std::vector<T> values, double mean)
{
    double variance = 0.0f;
    for (auto v : values)
    {
        const double x = static_cast<double>(v) - mean;
        variance += x * x;
    }
    return variance / values.size();
}

class poisson_distribution_tests : public ::testing::TestWithParam<double> { };

TEST_P(poisson_distribution_tests, mean_var)
{
    const double lambda = GetParam();

    std::random_device rd;
    std::mt19937 gen(rd());

    rocrand_poisson_distribution<ROCRAND_DISCRETE_METHOD_ALIAS, true> dis;
    dis.set_lambda(lambda);

    const size_t samples_count = static_cast<size_t>(std::max(2.0, sqrt(lambda))) * 100000;
    std::vector<unsigned int> values(samples_count);

    for (size_t si = 0; si < samples_count; si++)
    {
        const unsigned int v = dis(gen());
        values[si] = v;
    }

    const double mean = get_mean(values);
    const double variance = get_variance(values, mean);

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-2));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-2));
}

TEST_P(poisson_distribution_tests, histogram_compare)
{
    const double lambda = GetParam();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<unsigned int> host_dis(lambda);

    rocrand_poisson_distribution<ROCRAND_DISCRETE_METHOD_ALIAS, true> dis;
    dis.set_lambda(lambda);

    const size_t samples_count = static_cast<size_t>(std::max(2.0, sqrt(lambda))) * 100000;
    const size_t bin_size = static_cast<size_t>(std::max(2.0, sqrt(lambda)));
    const size_t bins_count = static_cast<size_t>((2.0 * lambda + 10.0) / bin_size);
    std::vector<unsigned int> historgram0(bins_count);
    std::vector<unsigned int> historgram1(bins_count);

    for (size_t si = 0; si < samples_count; si++)
    {
        const unsigned int v = host_dis(gen);
        const size_t bin = v / bin_size;
        if (bin < bins_count)
        {
            historgram0[bin]++;
        }
    }

    for (size_t si = 0; si < samples_count; si++)
    {
        const unsigned int v = dis(gen());
        const size_t bin = v / bin_size;
        if (bin < bins_count)
        {
            historgram1[bin]++;
        }
    }

    // Very loose comparison
    for (size_t bi = 0; bi < bins_count; bi++)
    {
        const unsigned int h0 = historgram0[bi];
        const unsigned int h1 = historgram1[bi];
        EXPECT_NEAR(h0, h1, std::max(samples_count * 1e-3, std::max(h0, h1) * 1e-1));
    }
}

const double lambdas[] = { 1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0 };

INSTANTIATE_TEST_CASE_P(poisson_distribution_tests,
                        poisson_distribution_tests,
                        ::testing::ValuesIn(lambdas));
