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

#include <gtest/gtest.h>
#include <stdio.h>

#include <hip/hip_runtime.h>
#include <rocrand.hpp>

#define HIP_CHECK(x) ASSERT_EQ(x, hipSuccess)

TEST(rocrand_cpp_wrapper, rocrand_error)
{
    rocrand_cpp::error e(ROCRAND_STATUS_SUCCESS);
    EXPECT_EQ(e.error_code(), ROCRAND_STATUS_SUCCESS);
}

template <class T>
void rocrand_rng_ctor_template()
{
    rocrand_generator generator = NULL;
    ASSERT_EQ(rocrand_create_generator(&generator, T::type()), ROCRAND_STATUS_SUCCESS);
    ASSERT_NE(generator, (rocrand_generator)NULL);
    T x(generator);
    ASSERT_EQ(generator, (rocrand_generator)NULL);

    try
    {
        T y(generator);
        FAIL() << "Expected rocrand_cpp::error";
    }
    catch(const rocrand_cpp::error& err)
    {
        EXPECT_EQ(err.error_code(), ROCRAND_STATUS_NOT_CREATED);
    }
    catch(...)
    {
        FAIL() << "Expected rocrand_cpp::error";
    }
}

TEST(rocrand_cpp_wrapper, rocrand_rng_ctor)
{
    ASSERT_NO_THROW(rocrand_rng_ctor_template<rocrand_cpp::philox4x32_10>());
    ASSERT_NO_THROW(rocrand_rng_ctor_template<rocrand_cpp::xorwow>());
    ASSERT_NO_THROW(rocrand_rng_ctor_template<rocrand_cpp::mrg32k3a>());
    ASSERT_NO_THROW(rocrand_rng_ctor_template<rocrand_cpp::mtgp32>());
    ASSERT_NO_THROW(rocrand_rng_ctor_template<rocrand_cpp::sobol32>());
}

template <class T>
void rocrand_prng_ctor_template()
{
    T();
    T(11ULL); // seed
    T(11ULL, 2ULL); // seed, offset

    rocrand_cpp::random_device rd;
    T(rd(), 2ULL); // seed, offset
}

TEST(rocrand_cpp_wrapper, rocrand_prng_ctor)
{
    ASSERT_NO_THROW(rocrand_prng_ctor_template<rocrand_cpp::philox4x32_10>());
    ASSERT_NO_THROW(rocrand_prng_ctor_template<rocrand_cpp::xorwow>());
    ASSERT_NO_THROW(rocrand_prng_ctor_template<rocrand_cpp::mrg32k3a>());

    // mtgp32 does not have ctor with offset
    rocrand_cpp::mtgp32();
    rocrand_cpp::mtgp32(11ULL);
}

template <class T1, class T2>
void assert_same_types()
{
    ::testing::StaticAssertTypeEq<T1, T2>();
}

TEST(rocrand_cpp_wrapper, rocrand_rng_result_type)
{
    assert_same_types<unsigned int, rocrand_cpp::philox4x32_10::result_type>();
    assert_same_types<unsigned int, rocrand_cpp::xorwow::result_type>();
    assert_same_types<unsigned int, rocrand_cpp::mrg32k3a::result_type>();
    assert_same_types<unsigned int, rocrand_cpp::mtgp32::result_type>();
    assert_same_types<unsigned int, rocrand_cpp::sobol32::result_type>();
}

TEST(rocrand_cpp_wrapper, rocrand_rng_offset_type)
{
    assert_same_types<unsigned long long, rocrand_cpp::philox4x32_10::offset_type>();
    assert_same_types<unsigned long long, rocrand_cpp::xorwow::offset_type>();
    assert_same_types<unsigned long long, rocrand_cpp::mrg32k3a::offset_type>();
    assert_same_types<unsigned long long, rocrand_cpp::mtgp32::offset_type>();
    assert_same_types<unsigned long long, rocrand_cpp::sobol32::offset_type>();
}

TEST(rocrand_cpp_wrapper, rocrand_prng_default_seed)
{
    EXPECT_EQ(rocrand_cpp::philox4x32_10::default_seed, ROCRAND_PHILOX4x32_DEFAULT_SEED);
    EXPECT_EQ(rocrand_cpp::xorwow::default_seed, ROCRAND_XORWOW_DEFAULT_SEED);
    EXPECT_EQ(rocrand_cpp::mrg32k3a::default_seed, ROCRAND_MRG32K3A_DEFAULT_SEED);
}

TEST(rocrand_cpp_wrapper, rocrand_qrng_default_num_dimensions)
{
    EXPECT_EQ(rocrand_cpp::sobol32::default_num_dimensions, 1U);
}

template <class T>
void rocrand_qrng_ctor_template()
{
    T();
    T(11U); // dimensions
    T(11U, 2ULL); // dimensions, offset
    T(20000, 2ULL); // dimensions, offset

    try
    {
        T(20001, 2ULL);
        FAIL() << "Expected rocrand_cpp::error";
    }
    catch(const rocrand_cpp::error& err)
    {
        EXPECT_EQ(err.error_code(), ROCRAND_STATUS_OUT_OF_RANGE);
    }
    catch(...)
    {
        FAIL() << "Expected rocrand_cpp::error";
    }
}

TEST(rocrand_cpp_wrapper, rocrand_qrng_ctor)
{
    ASSERT_NO_THROW(rocrand_qrng_ctor_template<rocrand_cpp::sobol32>());
}

template <class T>
void rocrand_prng_seed_template()
{
    T engine;
    engine.seed(11ULL);
    engine.seed(12ULL);
}

TEST(rocrand_cpp_wrapper, rocrand_prng_seed)
{
    ASSERT_NO_THROW(rocrand_prng_seed_template<rocrand_cpp::philox4x32_10>());
    ASSERT_NO_THROW(rocrand_prng_seed_template<rocrand_cpp::xorwow>());
    ASSERT_NO_THROW(rocrand_prng_seed_template<rocrand_cpp::mrg32k3a>());
    ASSERT_NO_THROW(rocrand_prng_seed_template<rocrand_cpp::mtgp32>());
}

template <class T>
void rocrand_qrng_dims_template()
{
    T engine;
    engine.dimensions(11U);
    engine.dimensions(20000U);

    try
    {
        engine.dimensions(20001U);
        FAIL() << "Expected rocrand_cpp::error";
    }
    catch(const rocrand_cpp::error& err)
    {
        EXPECT_EQ(err.error_code(), ROCRAND_STATUS_OUT_OF_RANGE);
    }
    catch(...)
    {
        FAIL() << "Expected rocrand_cpp::error";
    }
}

TEST(rocrand_cpp_wrapper, rocrand_qrng_dims)
{
    ASSERT_NO_THROW(rocrand_qrng_dims_template<rocrand_cpp::sobol32>());
}

template <class T>
void rocrand_rng_offset_template()
{
    T engine;
    engine.offset(11ULL);
    engine.offset(12ULL);
}

TEST(rocrand_cpp_wrapper, rocrand_rng_offset)
{
    ASSERT_NO_THROW(rocrand_rng_offset_template<rocrand_cpp::philox4x32_10>());
    ASSERT_NO_THROW(rocrand_rng_offset_template<rocrand_cpp::xorwow>());
    ASSERT_NO_THROW(rocrand_rng_offset_template<rocrand_cpp::mrg32k3a>());
    ASSERT_NO_THROW(rocrand_rng_offset_template<rocrand_cpp::sobol32>());
}

template <class T>
void rocrand_rng_stream_template()
{
    T           engine;
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    engine.stream(stream);
    engine.stream(NULL);
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST(rocrand_cpp_wrapper, rocrand_rng_stream)
{
    ASSERT_NO_THROW(rocrand_rng_stream_template<rocrand_cpp::philox4x32_10>());
    ASSERT_NO_THROW(rocrand_rng_stream_template<rocrand_cpp::xorwow>());
    ASSERT_NO_THROW(rocrand_rng_stream_template<rocrand_cpp::mrg32k3a>());
    ASSERT_NO_THROW(rocrand_rng_stream_template<rocrand_cpp::mtgp32>());
    ASSERT_NO_THROW(rocrand_rng_stream_template<rocrand_cpp::sobol32>());
}

template <class T, class IntType>
void rocrand_uniform_int_dist_template()
{
    T                                              engine;
    rocrand_cpp::uniform_int_distribution<IntType> d;

    const size_t output_size = 8192;
    IntType*     output;
    HIP_CHECK(hipMalloc((void**)&output, output_size * sizeof(IntType)));
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> output_host(output_size);
    HIP_CHECK(hipMemcpy(
        output_host.data(), output, output_size * sizeof(unsigned int), hipMemcpyDeviceToHost));
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

TEST(rocrand_cpp_wrapper, rocrand_uniform_int_dist)
{
    ASSERT_NO_THROW(
        (rocrand_uniform_int_dist_template<rocrand_cpp::philox4x32_10, unsigned int>()));
    ASSERT_NO_THROW((rocrand_uniform_int_dist_template<rocrand_cpp::xorwow, unsigned int>()));
    ASSERT_NO_THROW((rocrand_uniform_int_dist_template<rocrand_cpp::mrg32k3a, unsigned int>()));
    ASSERT_NO_THROW((rocrand_uniform_int_dist_template<rocrand_cpp::mtgp32, unsigned int>()));
    ASSERT_NO_THROW((rocrand_uniform_int_dist_template<rocrand_cpp::sobol32, unsigned int>()));
}

template <class T, class RealType>
void rocrand_uniform_real_dist_template()
{
    T                                                engine;
    rocrand_cpp::uniform_real_distribution<RealType> d;

    const size_t output_size = 8192;
    RealType*    output;
    HIP_CHECK(hipMalloc((void**)&output, output_size * sizeof(RealType)));
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<RealType> output_host(output_size);
    HIP_CHECK(hipMemcpy(
        output_host.data(), output, output_size * sizeof(RealType), hipMemcpyDeviceToHost));
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

TEST(rocrand_cpp_wrapper, rocrand_uniform_real_dist_float)
{
    ASSERT_NO_THROW((rocrand_uniform_real_dist_template<rocrand_cpp::philox4x32_10, float>()));
    ASSERT_NO_THROW((rocrand_uniform_real_dist_template<rocrand_cpp::xorwow, float>()));
    ASSERT_NO_THROW((rocrand_uniform_real_dist_template<rocrand_cpp::mrg32k3a, float>()));
    ASSERT_NO_THROW((rocrand_uniform_real_dist_template<rocrand_cpp::mtgp32, float>()));
    ASSERT_NO_THROW((rocrand_uniform_real_dist_template<rocrand_cpp::sobol32, float>()));
}

TEST(rocrand_cpp_wrapper, rocrand_uniform_real_dist_double)
{
    ASSERT_NO_THROW((rocrand_uniform_real_dist_template<rocrand_cpp::philox4x32_10, double>()));
    ASSERT_NO_THROW((rocrand_uniform_real_dist_template<rocrand_cpp::xorwow, double>()));
    ASSERT_NO_THROW((rocrand_uniform_real_dist_template<rocrand_cpp::mrg32k3a, double>()));
    ASSERT_NO_THROW((rocrand_uniform_real_dist_template<rocrand_cpp::mtgp32, double>()));
    ASSERT_NO_THROW((rocrand_uniform_real_dist_template<rocrand_cpp::sobol32, double>()));
}

template <class T, class RealType>
void rocrand_normal_dist_template()
{
    T                                          engine;
    rocrand_cpp::normal_distribution<RealType> d;

    const size_t output_size = 8192;
    RealType*    output;
    HIP_CHECK(hipMalloc((void**)&output, output_size * sizeof(RealType)));
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<RealType> output_host(output_size);
    HIP_CHECK(hipMemcpy(
        output_host.data(), output, output_size * sizeof(RealType), hipMemcpyDeviceToHost));
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

TEST(rocrand_cpp_wrapper, rocrand_normal_dist_float)
{
    ASSERT_NO_THROW((rocrand_normal_dist_template<rocrand_cpp::philox4x32_10, float>()));
    ASSERT_NO_THROW((rocrand_normal_dist_template<rocrand_cpp::xorwow, float>()));
    ASSERT_NO_THROW((rocrand_normal_dist_template<rocrand_cpp::mrg32k3a, float>()));
    ASSERT_NO_THROW((rocrand_normal_dist_template<rocrand_cpp::mtgp32, float>()));
    ASSERT_NO_THROW((rocrand_normal_dist_template<rocrand_cpp::sobol32, float>()));
}

TEST(rocrand_cpp_wrapper, rocrand_normal_dist_double)
{
    ASSERT_NO_THROW((rocrand_normal_dist_template<rocrand_cpp::philox4x32_10, double>()));
    ASSERT_NO_THROW((rocrand_normal_dist_template<rocrand_cpp::xorwow, double>()));
    ASSERT_NO_THROW((rocrand_normal_dist_template<rocrand_cpp::mrg32k3a, double>()));
    ASSERT_NO_THROW((rocrand_normal_dist_template<rocrand_cpp::mtgp32, double>()));
    ASSERT_NO_THROW((rocrand_normal_dist_template<rocrand_cpp::sobol32, double>()));
}

TEST(rocrand_cpp_wrapper, rocrand_normal_dist_param)
{
    rocrand_cpp::normal_distribution<> d1(1.0f, 3.0f);
    rocrand_cpp::normal_distribution<> d2(1.0f, 3.0f);
    rocrand_cpp::normal_distribution<> d3(2.0f, 4.0f);

    ASSERT_TRUE(d1.param() == d2.param());
    ASSERT_TRUE(d1.param() == d1.param());
    ASSERT_TRUE(d1.param() != d3.param());

    d3.param(d1.param());
    ASSERT_TRUE(d1.param() == d3.param());
}

template <class T, class RealType>
void rocrand_lognormal_dist_template()
{
    T                                             engine;
    rocrand_cpp::lognormal_distribution<RealType> d(1.6, 0.25);

    const size_t output_size = 8192;
    RealType*    output;
    HIP_CHECK(hipMalloc((void**)&output, output_size * sizeof(RealType)));
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<RealType> output_host(output_size);
    HIP_CHECK(hipMemcpy(
        output_host.data(), output, output_size * sizeof(RealType), hipMemcpyDeviceToHost));
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
    double logstd  = std::sqrt(std::log(1.0f + stddev / (mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);
}

TEST(rocrand_cpp_wrapper, rocrand_lognormal_dist_float)
{
    ASSERT_NO_THROW((rocrand_lognormal_dist_template<rocrand_cpp::philox4x32_10, float>()));
    ASSERT_NO_THROW((rocrand_lognormal_dist_template<rocrand_cpp::xorwow, float>()));
    ASSERT_NO_THROW((rocrand_lognormal_dist_template<rocrand_cpp::mrg32k3a, float>()));
    ASSERT_NO_THROW((rocrand_lognormal_dist_template<rocrand_cpp::mtgp32, float>()));
    ASSERT_NO_THROW((rocrand_lognormal_dist_template<rocrand_cpp::sobol32, float>()));
}

TEST(rocrand_cpp_wrapper, rocrand_lognormal_dist_double)
{
    ASSERT_NO_THROW((rocrand_lognormal_dist_template<rocrand_cpp::philox4x32_10, double>()));
    ASSERT_NO_THROW((rocrand_lognormal_dist_template<rocrand_cpp::xorwow, double>()));
    ASSERT_NO_THROW((rocrand_lognormal_dist_template<rocrand_cpp::mrg32k3a, double>()));
    ASSERT_NO_THROW((rocrand_lognormal_dist_template<rocrand_cpp::mtgp32, double>()));
    ASSERT_NO_THROW((rocrand_lognormal_dist_template<rocrand_cpp::sobol32, double>()));
}

TEST(rocrand_cpp_wrapper, rocrand_lognormal_dist_param)
{
    rocrand_cpp::lognormal_distribution<> d1(1.0f, 3.0f);
    rocrand_cpp::lognormal_distribution<> d2(1.0f, 3.0f);
    rocrand_cpp::lognormal_distribution<> d3(2.0f, 4.0f);

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

template <class T, class IntType>
void rocrand_poisson_dist_template(const double lambda)
{
    T                                          engine;
    rocrand_cpp::poisson_distribution<IntType> d(lambda);

    const size_t output_size = 8192;
    IntType*     output;
    HIP_CHECK(hipMalloc((void**)&output, output_size * sizeof(IntType)));
    HIP_CHECK(hipDeviceSynchronize());

    // generate
    EXPECT_NO_THROW(d(engine, output, output_size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<IntType> output_host(output_size);
    HIP_CHECK(hipMemcpy(
        output_host.data(), output, output_size * sizeof(IntType), hipMemcpyDeviceToHost));
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

class poisson_dist : public ::testing::TestWithParam<double>
{
};

TEST_P(poisson_dist, rocrand_poisson_dist)
{
    const double lambda = GetParam();
    ASSERT_NO_THROW(
        (rocrand_poisson_dist_template<rocrand_cpp::philox4x32_10, unsigned int>(lambda)));
    ASSERT_NO_THROW((rocrand_poisson_dist_template<rocrand_cpp::xorwow, unsigned int>(lambda)));
    ASSERT_NO_THROW((rocrand_poisson_dist_template<rocrand_cpp::mrg32k3a, unsigned int>(lambda)));
    ASSERT_NO_THROW((rocrand_poisson_dist_template<rocrand_cpp::mtgp32, unsigned int>(lambda)));
    ASSERT_NO_THROW((rocrand_poisson_dist_template<rocrand_cpp::sobol32, unsigned int>(lambda)));
}

const double lambdas[] = {1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0};

INSTANTIATE_TEST_SUITE_P(rocrand_cpp_wrapper, poisson_dist, ::testing::ValuesIn(lambdas));

TEST(rocrand_cpp_wrapper, rocrand_poisson_dist_param)
{
    rocrand_cpp::poisson_distribution<> d1(1.0);
    rocrand_cpp::poisson_distribution<> d2(1.0);
    rocrand_cpp::poisson_distribution<> d3(2.0);

    ASSERT_TRUE(d1.mean() == d1.param().mean());
    ASSERT_TRUE(d1.mean() == 1.0);

    ASSERT_TRUE(d1.param() == d2.param());
    ASSERT_TRUE(d1.param() == d1.param());
    ASSERT_TRUE(d1.param() != d3.param());

    d3.param(d1.param());
    ASSERT_TRUE(d1.param() == d3.param());
}
