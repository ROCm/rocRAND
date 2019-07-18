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
#include <hiprand.hpp>

#define HIP_CHECK(x) ASSERT_EQ(x, hipSuccess)

TEST(hiprand_cpp_wrapper, hiprand_error)
{
    hiprand_cpp::error e(HIPRAND_STATUS_SUCCESS);
    EXPECT_EQ(e.error_code(), HIPRAND_STATUS_SUCCESS);
}

template<class T>
void hiprand_rng_ctor_template()
{
    hiprandGenerator_t generator = NULL;
    ASSERT_EQ(hiprandCreateGenerator(&generator, T::type()), HIPRAND_STATUS_SUCCESS);
    ASSERT_NE(generator, (hiprandGenerator_t)NULL);
    T x(generator);
    ASSERT_EQ(generator, (hiprandGenerator_t)NULL);

    try {
        T y(generator);
        FAIL() << "Expected hiprand_cpp::error";
    }
    catch(const hiprand_cpp::error& err) {
        EXPECT_EQ(err.error_code(), HIPRAND_STATUS_NOT_INITIALIZED);
    }
    catch(...) {
        FAIL() << "Expected hiprand_cpp::error";
    }
}

TEST(hiprand_cpp_wrapper, hiprand_rng_ctor)
{
    ASSERT_NO_THROW(hiprand_rng_ctor_template<hiprand_cpp::philox4x32_10>());
    ASSERT_NO_THROW(hiprand_rng_ctor_template<hiprand_cpp::xorwow>());
    ASSERT_NO_THROW(hiprand_rng_ctor_template<hiprand_cpp::mrg32k3a>());
    ASSERT_NO_THROW(hiprand_rng_ctor_template<hiprand_cpp::mtgp32>());
    ASSERT_NO_THROW(hiprand_rng_ctor_template<hiprand_cpp::sobol32>());
}

template<class T>
void hiprand_prng_ctor_template()
{
    T();
    T(11ULL); // seed
    T(11ULL, 2ULL); // seed, offset

    hiprand_cpp::random_device rd;
    T(rd(), 2ULL); // seed, offset
}

TEST(hiprand_cpp_wrapper, hiprand_prng_ctor)
{
    ASSERT_NO_THROW(hiprand_prng_ctor_template<hiprand_cpp::philox4x32_10>());
    ASSERT_NO_THROW(hiprand_prng_ctor_template<hiprand_cpp::xorwow>());
    ASSERT_NO_THROW(hiprand_prng_ctor_template<hiprand_cpp::mrg32k3a>());

    // mtgp32 does not have ctor with offset
    hiprand_cpp::mtgp32();
    hiprand_cpp::mtgp32(11ULL);
}

template<class T1, class T2>
void assert_same_types()
{
    ::testing::StaticAssertTypeEq<T1, T2>();
}

TEST(hiprand_cpp_wrapper, hiprand_rng_result_type)
{
    assert_same_types<unsigned int, hiprand_cpp::philox4x32_10::result_type>();
    assert_same_types<unsigned int, hiprand_cpp::xorwow::result_type>();
    assert_same_types<unsigned int, hiprand_cpp::mrg32k3a::result_type>();
    assert_same_types<unsigned int, hiprand_cpp::mtgp32::result_type>();
    assert_same_types<unsigned int, hiprand_cpp::sobol32::result_type>();
}

TEST(hiprand_cpp_wrapper, hiprand_rng_offset_type)
{
    assert_same_types<unsigned long long, hiprand_cpp::philox4x32_10::offset_type>();
    assert_same_types<unsigned long long, hiprand_cpp::xorwow::offset_type>();
    assert_same_types<unsigned long long, hiprand_cpp::mrg32k3a::offset_type>();
    assert_same_types<unsigned long long, hiprand_cpp::mtgp32::offset_type>();
    assert_same_types<unsigned long long, hiprand_cpp::sobol32::offset_type>();
}

TEST(hiprand_cpp_wrapper, hiprand_prng_default_seed)
{
    EXPECT_EQ(hiprand_cpp::philox4x32_10::default_seed, HIPRAND_PHILOX4x32_DEFAULT_SEED);
    EXPECT_EQ(hiprand_cpp::xorwow::default_seed, HIPRAND_XORWOW_DEFAULT_SEED);
    EXPECT_EQ(hiprand_cpp::mrg32k3a::default_seed, HIPRAND_MRG32K3A_DEFAULT_SEED);
}

TEST(hiprand_cpp_wrapper, hiprand_qrng_default_num_dimensions)
{
    EXPECT_EQ(hiprand_cpp::sobol32::default_num_dimensions, 1U);
}

template<class T>
void hiprand_qrng_ctor_template()
{
    T();
    T(11U); // dimensions
    T(11U, 2ULL); // dimensions, offset
    T(20000, 2ULL); // dimensions, offset

    try {
        T(20001, 2ULL);
        FAIL() << "Expected hiprand_cpp::error";
    }
    catch(const hiprand_cpp::error& err) {
        EXPECT_EQ(err.error_code(), HIPRAND_STATUS_OUT_OF_RANGE);
    }
    catch(...) {
        FAIL() << "Expected hiprand_cpp::error";
    }
}

TEST(hiprand_cpp_wrapper, hiprand_qrng_ctor)
{
    ASSERT_NO_THROW(hiprand_qrng_ctor_template<hiprand_cpp::sobol32>());
}

template<class T>
void hiprand_prng_seed_template()
{
    T engine;
    engine.seed(11ULL);
    engine.seed(12ULL);
}

TEST(hiprand_cpp_wrapper, hiprand_prng_seed)
{
    ASSERT_NO_THROW(hiprand_prng_seed_template<hiprand_cpp::philox4x32_10>());
    ASSERT_NO_THROW(hiprand_prng_seed_template<hiprand_cpp::xorwow>());
    ASSERT_NO_THROW(hiprand_prng_seed_template<hiprand_cpp::mrg32k3a>());
    ASSERT_NO_THROW(hiprand_prng_seed_template<hiprand_cpp::mtgp32>());
}

template<class T>
void hiprand_qrng_dims_template()
{
    T engine;
    engine.dimensions(11U);
    engine.dimensions(20000U);

    try {
        engine.dimensions(20001U);
        FAIL() << "Expected hiprand_cpp::error";
    }
    catch(const hiprand_cpp::error& err) {
        EXPECT_EQ(err.error_code(), HIPRAND_STATUS_OUT_OF_RANGE);
    }
    catch(...) {
        FAIL() << "Expected hiprand_cpp::error";
    }
}

TEST(hiprand_cpp_wrapper, hiprand_qrng_dims)
{
    ASSERT_NO_THROW(hiprand_qrng_dims_template<hiprand_cpp::sobol32>());
}

template<class T>
void hiprand_rng_offset_template()
{
    T engine;
    engine.offset(11ULL);
    engine.offset(12ULL);
}

TEST(hiprand_cpp_wrapper, hiprand_rng_offset)
{
    ASSERT_NO_THROW(hiprand_rng_offset_template<hiprand_cpp::philox4x32_10>());
    ASSERT_NO_THROW(hiprand_rng_offset_template<hiprand_cpp::xorwow>());
    ASSERT_NO_THROW(hiprand_rng_offset_template<hiprand_cpp::mrg32k3a>());
    ASSERT_NO_THROW(hiprand_rng_offset_template<hiprand_cpp::sobol32>());
}

template<class T>
void hiprand_rng_stream_template()
{
    T engine;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    engine.stream(stream);
    engine.stream(NULL);
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST(hiprand_cpp_wrapper, hiprand_rng_stream)
{
    ASSERT_NO_THROW(hiprand_rng_stream_template<hiprand_cpp::philox4x32_10>());
    ASSERT_NO_THROW(hiprand_rng_stream_template<hiprand_cpp::xorwow>());
    ASSERT_NO_THROW(hiprand_rng_stream_template<hiprand_cpp::mrg32k3a>());
    ASSERT_NO_THROW(hiprand_rng_stream_template<hiprand_cpp::mtgp32>());
    ASSERT_NO_THROW(hiprand_rng_stream_template<hiprand_cpp::sobol32>());
}

template<class T, class IntType>
void hiprand_uniform_int_dist_template()
{
    T engine;
    hiprand_cpp::uniform_int_distribution<IntType> d;

    const size_t output_size = 8192;
    IntType * output;
    HIP_CHECK(
        hipMalloc((void **)&output,
        output_size * sizeof(IntType))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
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
}

TEST(hiprand_cpp_wrapper, hiprand_uniform_int_dist)
{
    ASSERT_NO_THROW((
        hiprand_uniform_int_dist_template<hiprand_cpp::philox4x32_10, unsigned int>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_int_dist_template<hiprand_cpp::xorwow, unsigned int>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_int_dist_template<hiprand_cpp::mrg32k3a, unsigned int>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_int_dist_template<hiprand_cpp::mtgp32, unsigned int>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_int_dist_template<hiprand_cpp::sobol32, unsigned int>()
    ));
}

template<class T, class RealType>
void hiprand_uniform_real_dist_template()
{
    T engine;
    hiprand_cpp::uniform_real_distribution<RealType> d;

    const size_t output_size = 8192;
    RealType * output;
    HIP_CHECK(
        hipMalloc((void **)&output,
        output_size * sizeof(RealType))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<RealType> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(RealType),
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
}

TEST(hiprand_cpp_wrapper, hiprand_uniform_real_dist_float)
{
    ASSERT_NO_THROW((
        hiprand_uniform_real_dist_template<hiprand_cpp::philox4x32_10, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_real_dist_template<hiprand_cpp::xorwow, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_real_dist_template<hiprand_cpp::mrg32k3a, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_real_dist_template<hiprand_cpp::mtgp32, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_real_dist_template<hiprand_cpp::sobol32, float>()
    ));
}

TEST(hiprand_cpp_wrapper, hiprand_uniform_real_dist_double)
{
    ASSERT_NO_THROW((
        hiprand_uniform_real_dist_template<hiprand_cpp::philox4x32_10, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_real_dist_template<hiprand_cpp::xorwow, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_real_dist_template<hiprand_cpp::mrg32k3a, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_real_dist_template<hiprand_cpp::mtgp32, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_uniform_real_dist_template<hiprand_cpp::sobol32, double>()
    ));
}

template<class T, class RealType>
void hiprand_normal_dist_template()
{
    T engine;
    hiprand_cpp::normal_distribution<RealType> d;

    const size_t output_size = 8192;
    RealType * output;
    HIP_CHECK(
        hipMalloc((void **)&output,
        output_size * sizeof(RealType))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<RealType> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(RealType),
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
}

TEST(hiprand_cpp_wrapper, hiprand_normal_dist_float)
{
    ASSERT_NO_THROW((
        hiprand_normal_dist_template<hiprand_cpp::philox4x32_10, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_normal_dist_template<hiprand_cpp::xorwow, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_normal_dist_template<hiprand_cpp::mrg32k3a, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_normal_dist_template<hiprand_cpp::mtgp32, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_normal_dist_template<hiprand_cpp::sobol32, float>()
    ));
}

TEST(hiprand_cpp_wrapper, hiprand_normal_dist_double)
{
    ASSERT_NO_THROW((
        hiprand_normal_dist_template<hiprand_cpp::philox4x32_10, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_normal_dist_template<hiprand_cpp::xorwow, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_normal_dist_template<hiprand_cpp::mrg32k3a, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_normal_dist_template<hiprand_cpp::mtgp32, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_normal_dist_template<hiprand_cpp::sobol32, double>()
    ));
}

TEST(hiprand_cpp_wrapper, hiprand_normal_dist_param)
{
    hiprand_cpp::normal_distribution<> d1(1.0f, 3.0f);
    hiprand_cpp::normal_distribution<> d2(1.0f, 3.0f);
    hiprand_cpp::normal_distribution<> d3(2.0f, 4.0f);

    ASSERT_TRUE(d1.param() == d2.param());
    ASSERT_TRUE(d1.param() == d1.param());
    ASSERT_TRUE(d1.param() != d3.param());

    d3.param(d1.param());
    ASSERT_TRUE(d1.param() == d3.param());
}

template<class T, class RealType>
void hiprand_lognormal_dist_template()
{
    T engine;
    hiprand_cpp::lognormal_distribution<RealType> d(1.6, 0.25);

    const size_t output_size = 8192;
    RealType * output;
    HIP_CHECK(
        hipMalloc((void **)&output,
        output_size * sizeof(RealType))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<RealType> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(RealType),
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
}

TEST(hiprand_cpp_wrapper, hiprand_lognormal_dist_float)
{
    ASSERT_NO_THROW((
        hiprand_lognormal_dist_template<hiprand_cpp::philox4x32_10, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_lognormal_dist_template<hiprand_cpp::xorwow, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_lognormal_dist_template<hiprand_cpp::mrg32k3a, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_lognormal_dist_template<hiprand_cpp::mtgp32, float>()
    ));
    ASSERT_NO_THROW((
        hiprand_lognormal_dist_template<hiprand_cpp::sobol32, float>()
    ));
}

TEST(hiprand_cpp_wrapper, hiprand_lognormal_dist_double)
{
    ASSERT_NO_THROW((
        hiprand_lognormal_dist_template<hiprand_cpp::philox4x32_10, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_lognormal_dist_template<hiprand_cpp::xorwow, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_lognormal_dist_template<hiprand_cpp::mrg32k3a, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_lognormal_dist_template<hiprand_cpp::mtgp32, double>()
    ));
    ASSERT_NO_THROW((
        hiprand_lognormal_dist_template<hiprand_cpp::sobol32, double>()
    ));
}

TEST(hiprand_cpp_wrapper, hiprand_lognormal_dist_param)
{
    hiprand_cpp::lognormal_distribution<> d1(1.0f, 3.0f);
    hiprand_cpp::lognormal_distribution<> d2(1.0f, 3.0f);
    hiprand_cpp::lognormal_distribution<> d3(2.0f, 4.0f);

    ASSERT_TRUE(d1.m() == d1.param().m());
    ASSERT_TRUE(d1.m() == 1.0f);
    ASSERT_TRUE(d1.s() == d1.param().s());
    ASSERT_TRUE(d1.s() == 3.0f);

    ASSERT_TRUE(d1.param() == d2.param());
    ASSERT_TRUE(d1.param() == d1.param());
    ASSERT_TRUE(d1.param() != d3.param());

    d3.param(d1.param());
    ASSERT_TRUE(d1.param() == d3.param());
}

template<class T, class IntType>
void hiprand_poisson_dist_template(const double lambda)
{
    T engine;
    hiprand_cpp::poisson_distribution<IntType> d(lambda);

    const size_t output_size = 8192;
    IntType * output;
    HIP_CHECK(
        hipMalloc((void **)&output,
        output_size * sizeof(IntType))
    );
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<IntType> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(IntType),
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
}

class poisson_dist : public ::testing::TestWithParam<double> { };

TEST_P(poisson_dist, hiprand_poisson_dist)
{
    const double lambda = GetParam();
    ASSERT_NO_THROW((
        hiprand_poisson_dist_template<hiprand_cpp::philox4x32_10, unsigned int>(lambda)
    ));
    ASSERT_NO_THROW((
        hiprand_poisson_dist_template<hiprand_cpp::xorwow, unsigned int>(lambda)
    ));
    ASSERT_NO_THROW((
        hiprand_poisson_dist_template<hiprand_cpp::mrg32k3a, unsigned int>(lambda)
    ));
    ASSERT_NO_THROW((
        hiprand_poisson_dist_template<hiprand_cpp::mtgp32, unsigned int>(lambda)
    ));
    ASSERT_NO_THROW((
        hiprand_poisson_dist_template<hiprand_cpp::sobol32, unsigned int>(lambda)
    ));
}

const double lambdas[] = { 1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0 };

INSTANTIATE_TEST_CASE_P(hiprand_cpp_wrapper,
                        poisson_dist,
                        ::testing::ValuesIn(lambdas));

TEST(hiprand_cpp_wrapper, hiprand_poisson_dist_param)
{
    hiprand_cpp::poisson_distribution<> d1(1.0);
    hiprand_cpp::poisson_distribution<> d2(1.0);
    hiprand_cpp::poisson_distribution<> d3(2.0);

    ASSERT_TRUE(d1.mean() == d1.param().mean());
    ASSERT_TRUE(d1.mean() == 1.0);

    ASSERT_TRUE(d1.param() == d2.param());
    ASSERT_TRUE(d1.param() == d1.param());
    ASSERT_TRUE(d1.param() != d3.param());

    d3.param(d1.param());
    ASSERT_TRUE(d1.param() == d3.param());
}
