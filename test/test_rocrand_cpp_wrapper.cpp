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
#include <rocrand.hpp>

#define HIP_CHECK(x) ASSERT_EQ(x, hipSuccess)

template<class T>
void rocrand_prng_ctor_template()
{
    T();
    T(11ULL); // seed
    T(11ULL, 2ULL); // seed, offset
}

TEST(rocrand_cpp_wrapper, rocrand_prng_ctor)
{
    ASSERT_NO_THROW(rocrand_prng_ctor_template<rocrand_cpp::philox4x32_10_engine>());
    ASSERT_NO_THROW(rocrand_prng_ctor_template<rocrand_cpp::xorwow_engine>());
    ASSERT_NO_THROW(rocrand_prng_ctor_template<rocrand_cpp::mrg32k3a_engine>());
}

template<class T>
void rocrand_prng_seed_template()
{
    T engine;
    engine.seed(11ULL);
    engine.seed(12ULL);
}

TEST(rocrand_cpp_wrapper, rocrand_prng_seed)
{
    ASSERT_NO_THROW(rocrand_prng_seed_template<rocrand_cpp::philox4x32_10_engine>());
    ASSERT_NO_THROW(rocrand_prng_seed_template<rocrand_cpp::xorwow_engine>());
    ASSERT_NO_THROW(rocrand_prng_seed_template<rocrand_cpp::mrg32k3a_engine>());
}

template<class T>
void rocrand_prng_offset_template()
{
    T engine;
    engine.offset(11ULL);
    engine.offset(12ULL);
}

TEST(rocrand_cpp_wrapper, rocrand_prng_offset)
{
    ASSERT_NO_THROW(rocrand_prng_offset_template<rocrand_cpp::philox4x32_10_engine>());
    ASSERT_NO_THROW(rocrand_prng_offset_template<rocrand_cpp::xorwow_engine>());
    ASSERT_NO_THROW(rocrand_prng_offset_template<rocrand_cpp::mrg32k3a_engine>());
}

template<class T>
void rocrand_rng_stream_template()
{
    T engine;
    hipStream_t stream;
    ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    engine.stream(stream);
    engine.stream(NULL);
}

TEST(rocrand_cpp_wrapper, rocrand_rng_stream)
{
    ASSERT_NO_THROW(rocrand_rng_stream_template<rocrand_cpp::philox4x32_10_engine>());
    ASSERT_NO_THROW(rocrand_rng_stream_template<rocrand_cpp::xorwow_engine>());
    ASSERT_NO_THROW(rocrand_rng_stream_template<rocrand_cpp::mrg32k3a_engine>());
}

template<class T, class IntType>
void rocrand_uniform_int_dist_template()
{
    T engine;
    rocrand_cpp::uniform_int_distribution<IntType> d;

    const size_t output_size = 8192;
    IntType * output;
    ASSERT_EQ(
        hipMalloc((void **)&output,
        output_size * sizeof(IntType)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

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

TEST(rocrand_cpp_wrapper, rocrand_uniform_int_dist)
{
    ASSERT_NO_THROW((
        rocrand_uniform_int_dist_template<rocrand_cpp::philox4x32_10_engine, unsigned int>()
    ));
    ASSERT_NO_THROW((
        rocrand_uniform_int_dist_template<rocrand_cpp::xorwow_engine, unsigned int>()
    ));
    ASSERT_NO_THROW((
        rocrand_uniform_int_dist_template<rocrand_cpp::mrg32k3a_engine, unsigned int>()
    ));
}

template<class T, class RealType>
void rocrand_uniform_real_dist_template()
{
    T engine;
    rocrand_cpp::uniform_real_distribution<RealType> d;

    const size_t output_size = 8192;
    RealType * output;
    ASSERT_EQ(
        hipMalloc((void **)&output,
        output_size * sizeof(RealType)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

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

TEST(rocrand_cpp_wrapper, rocrand_uniform_real_dist_float)
{
    ASSERT_NO_THROW((
        rocrand_uniform_real_dist_template<rocrand_cpp::philox4x32_10_engine, float>()
    ));
    ASSERT_NO_THROW((
        rocrand_uniform_real_dist_template<rocrand_cpp::xorwow_engine, float>()
    ));
    ASSERT_NO_THROW((
        rocrand_uniform_real_dist_template<rocrand_cpp::mrg32k3a_engine, float>()
    ));
}

TEST(rocrand_cpp_wrapper, rocrand_uniform_real_dist_double)
{
    ASSERT_NO_THROW((
        rocrand_uniform_real_dist_template<rocrand_cpp::philox4x32_10_engine, double>()
    ));
    ASSERT_NO_THROW((
        rocrand_uniform_real_dist_template<rocrand_cpp::xorwow_engine, double>()
    ));
    ASSERT_NO_THROW((
        rocrand_uniform_real_dist_template<rocrand_cpp::mrg32k3a_engine, double>()
    ));
}
