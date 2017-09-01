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

#include <hip/hip_runtime.h>
#include <hiprand.h>

#define HIP_CHECK(x) ASSERT_EQ(x, hipSuccess)
#define HIPRAND_CHECK(x) ASSERT_EQ(x, HIPRAND_STATUS_SUCCESS)

template<hiprandRngType_t rng_type>
void hiprand_generate_test_func()
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    unsigned int * output;
    ASSERT_EQ(
        hipMalloc((void **)&output,
        output_size * sizeof(unsigned int)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // generate
    HIPRAND_CHECK(hiprandGenerate(generator, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v) / UINT_MAX;
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST(hiprand, hiprand_generate_test)
{
    ASSERT_NO_THROW((
        hiprand_generate_test_func<HIPRAND_RNG_PSEUDO_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_test_func<HIPRAND_RNG_PSEUDO_XORWOW>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_test_func<HIPRAND_RNG_PSEUDO_MRG32K3A>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_test_func<HIPRAND_RNG_PSEUDO_MTGP32>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_test_func<HIPRAND_RNG_PSEUDO_PHILOX4_32_10>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_test_func<HIPRAND_RNG_QUASI_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_test_func<HIPRAND_RNG_QUASI_SOBOL32>()
    ));
}

template<hiprandRngType_t rng_type>
void hiprand_generate_uniform_test_func()
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    float * output;
    ASSERT_EQ(
        hipMalloc((void **)&output,
        output_size * sizeof(float)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // generate
    HIPRAND_CHECK(hiprandGenerateUniform(generator, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(float),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST(hiprand, hiprand_generate_uniform_test)
{
    ASSERT_NO_THROW((
        hiprand_generate_uniform_test_func<HIPRAND_RNG_PSEUDO_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_test_func<HIPRAND_RNG_PSEUDO_XORWOW>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_test_func<HIPRAND_RNG_PSEUDO_MRG32K3A>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_test_func<HIPRAND_RNG_PSEUDO_MTGP32>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_test_func<HIPRAND_RNG_PSEUDO_PHILOX4_32_10>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_test_func<HIPRAND_RNG_QUASI_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_test_func<HIPRAND_RNG_QUASI_SOBOL32>()
    ));
}

template<hiprandRngType_t rng_type>
void hiprand_generate_uniform_double_test_func()
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    double * output;
    ASSERT_EQ(
        hipMalloc((void **)&output,
        output_size * sizeof(double)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // generate
    HIPRAND_CHECK(hiprandGenerateUniformDouble(generator, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(double),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST(hiprand, hiprand_generate_uniform_double_test)
{
    ASSERT_NO_THROW((
        hiprand_generate_uniform_double_test_func<HIPRAND_RNG_PSEUDO_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_double_test_func<HIPRAND_RNG_PSEUDO_XORWOW>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_double_test_func<HIPRAND_RNG_PSEUDO_MRG32K3A>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_double_test_func<HIPRAND_RNG_PSEUDO_MTGP32>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_double_test_func<HIPRAND_RNG_PSEUDO_PHILOX4_32_10>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_double_test_func<HIPRAND_RNG_QUASI_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_uniform_double_test_func<HIPRAND_RNG_QUASI_SOBOL32>()
    ));
}

template<hiprandRngType_t rng_type>
void hiprand_generate_normal_test_func()
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    float * output;
    ASSERT_EQ(
        hipMalloc((void **)&output,
        output_size * sizeof(float)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // generate
    HIPRAND_CHECK(hiprandGenerateNormal(generator, output, output_size, 0, 1));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(float),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST(hiprand, hiprand_generate_normal_test)
{
    ASSERT_NO_THROW((
        hiprand_generate_normal_test_func<HIPRAND_RNG_PSEUDO_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_test_func<HIPRAND_RNG_PSEUDO_XORWOW>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_test_func<HIPRAND_RNG_PSEUDO_MRG32K3A>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_test_func<HIPRAND_RNG_PSEUDO_MTGP32>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_test_func<HIPRAND_RNG_PSEUDO_PHILOX4_32_10>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_test_func<HIPRAND_RNG_QUASI_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_test_func<HIPRAND_RNG_QUASI_SOBOL32>()
    ));
}

template<hiprandRngType_t rng_type>
void hiprand_generate_normal_double_test_func()
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    double * output;
    ASSERT_EQ(
        hipMalloc((void **)&output,
        output_size * sizeof(double)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // generate
    HIPRAND_CHECK(hiprandGenerateNormalDouble(generator, output, output_size, 0, 1));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(double),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<double>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST(hiprand, hiprand_generate_normal_double_test)
{
    ASSERT_NO_THROW((
        hiprand_generate_normal_double_test_func<HIPRAND_RNG_PSEUDO_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_double_test_func<HIPRAND_RNG_PSEUDO_XORWOW>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_double_test_func<HIPRAND_RNG_PSEUDO_MRG32K3A>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_double_test_func<HIPRAND_RNG_PSEUDO_MTGP32>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_double_test_func<HIPRAND_RNG_PSEUDO_PHILOX4_32_10>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_double_test_func<HIPRAND_RNG_QUASI_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_normal_double_test_func<HIPRAND_RNG_QUASI_SOBOL32>()
    ));
}

template<hiprandRngType_t rng_type>
void hiprand_generate_lognormal_test_func()
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    float * output;
    ASSERT_EQ(
        hipMalloc((void **)&output,
        output_size * sizeof(float)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // generate
    HIPRAND_CHECK(hiprandGenerateLogNormal(generator, output, output_size, 1.6f, 0.25f));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(float),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_size);

    double logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    double logstd = std::sqrt(std::log(1.0f + stddev/(mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST(hiprand, hiprand_generate_lognormal_test)
{
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_test_func<HIPRAND_RNG_PSEUDO_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_test_func<HIPRAND_RNG_PSEUDO_XORWOW>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_test_func<HIPRAND_RNG_PSEUDO_MRG32K3A>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_test_func<HIPRAND_RNG_PSEUDO_MTGP32>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_test_func<HIPRAND_RNG_PSEUDO_PHILOX4_32_10>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_test_func<HIPRAND_RNG_QUASI_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_test_func<HIPRAND_RNG_QUASI_SOBOL32>()
    ));
}

template<hiprandRngType_t rng_type>
void hiprand_generate_lognormal_double_test_func()
{
    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    double * output;
    ASSERT_EQ(
        hipMalloc((void **)&output,
        output_size * sizeof(double)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // generate
    HIPRAND_CHECK(hiprandGenerateLogNormalDouble(generator, output, output_size, 1.6, 0.25));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(double),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_size);

    double logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    double logstd = std::sqrt(std::log(1.0f + stddev/(mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST(hiprand, hiprand_generate_lognormal_double_test)
{
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_double_test_func<HIPRAND_RNG_PSEUDO_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_double_test_func<HIPRAND_RNG_PSEUDO_XORWOW>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_double_test_func<HIPRAND_RNG_PSEUDO_MRG32K3A>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_double_test_func<HIPRAND_RNG_PSEUDO_MTGP32>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_double_test_func<HIPRAND_RNG_PSEUDO_PHILOX4_32_10>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_double_test_func<HIPRAND_RNG_QUASI_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_lognormal_double_test_func<HIPRAND_RNG_QUASI_SOBOL32>()
    ));
}

template<hiprandRngType_t rng_type>
void hiprand_generate_poisson_test_func()
{
    double lambda = 20.0;

    hiprandGenerator_t generator = 0;
    HIPRAND_CHECK(hiprandCreateGenerator(&generator, rng_type));

    const size_t output_size = 8192;
    unsigned int * output;
    ASSERT_EQ(
        hipMalloc((void **)&output,
        output_size * sizeof(unsigned int)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // generate
    HIPRAND_CHECK(hiprandGeneratePoisson(generator, output, output_size, lambda));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;

    double variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));

    HIPRAND_CHECK(hiprandDestroyGenerator(generator));
}

TEST(hiprand, hiprand_generate_poisson_test)
{
    ASSERT_NO_THROW((
        hiprand_generate_poisson_test_func<HIPRAND_RNG_PSEUDO_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_poisson_test_func<HIPRAND_RNG_PSEUDO_XORWOW>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_poisson_test_func<HIPRAND_RNG_PSEUDO_MRG32K3A>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_poisson_test_func<HIPRAND_RNG_PSEUDO_MTGP32>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_poisson_test_func<HIPRAND_RNG_PSEUDO_PHILOX4_32_10>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_poisson_test_func<HIPRAND_RNG_QUASI_DEFAULT>()
    ));
    ASSERT_NO_THROW((
        hiprand_generate_poisson_test_func<HIPRAND_RNG_QUASI_SOBOL32>()
    ));
}
