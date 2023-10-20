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

#include <vector>

#define ROCRAND_CHECK(state) ASSERT_EQ(state, ROCRAND_STATUS_SUCCESS)

constexpr rocrand_rng_type host_rng_types[] = {};

class rocrand_generate_host_test : public ::testing::TestWithParam<rocrand_rng_type>
{};

TEST_P(rocrand_generate_host_test, int_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator_host(&generator, rng_type));

    std::vector<unsigned int> results(11111);

    ROCRAND_CHECK(rocrand_generate(generator, results.data(), results.size()));

    double mean = 0.0;
    for(unsigned int v : results)
    {
        mean += v;
    }

    mean /= static_cast<double>(std::numeric_limits<unsigned int>::max());
    mean /= results.size();

    ASSERT_NEAR(mean, 0.5, 0.05);

    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST_P(rocrand_generate_host_test, int_parity_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator device_generator, host_generator;
    ROCRAND_CHECK(rocrand_create_generator(&device_generator, rng_type));
    ROCRAND_CHECK(rocrand_create_generator_host(&host_generator, rng_type));

    std::vector<unsigned int> host_results(218192);
    ROCRAND_CHECK(rocrand_generate(host_generator, host_results.data(), host_results.size()));

    unsigned int* output;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&output),
                              host_results.size() * sizeof(unsigned int)));
    ROCRAND_CHECK(rocrand_generate(device_generator, output, host_results.size()));

    std::vector<unsigned int> device_results(host_results.size());
    HIP_CHECK(hipMemcpy(device_results.data(),
                        output,
                        host_results.size() * sizeof(unsigned int),
                        hipMemcpyDeviceToHost));

    ASSERT_EQ(host_results, device_results);

    ROCRAND_CHECK(rocrand_destroy_generator(host_generator));
    ROCRAND_CHECK(rocrand_destroy_generator(device_generator));
}

INSTANTIATE_TEST_SUITE_P(rocrand_generate_host_test,
                         rocrand_generate_host_test,
                         ::testing::ValuesIn(host_rng_types));
