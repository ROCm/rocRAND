// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "test_rocrand_common.hpp"

#include <rng/scrambled_sobol32.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>
#include <numeric>
#include <vector>

template<typename T>
struct rocrand_scrambled_sobol32_float_tests : public ::testing::Test
{
    using type = T;
};

using FloatReturnTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(rocrand_scrambled_sobol32_float_tests, FloatReturnTypes);

TYPED_TEST(rocrand_scrambled_sobol32_float_tests, uniform_test)
{
    using ResultType = typename TestFixture::type;

    constexpr size_t size = 1313;
    ResultType*      data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(ResultType) * size));

    rocrand_scrambled_sobol32 g;
    ROCRAND_CHECK(g.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<ResultType> host_data(size);
    HIP_CHECK(hipMemcpy(host_data.data(), data, sizeof(ResultType) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));

    double sum = 0;
    for(ResultType v : host_data)
    {
        ASSERT_GT(v, 0.0f);
        ASSERT_LE(v, 1.0f);
        sum += v;
    }
    const double mean = sum / size;

    ASSERT_NEAR(mean, 0.5f, 0.05f);
}

template<typename T>
struct rocrand_scrambled_sobol32_integer_tests : public ::testing::Test
{
    using type = T;
};

using UniformIntegerReturnTypes = ::testing::Types<unsigned int, unsigned char>;

TYPED_TEST_SUITE(rocrand_scrambled_sobol32_integer_tests, UniformIntegerReturnTypes);

TYPED_TEST(rocrand_scrambled_sobol32_integer_tests, uniform_test)
{
    using ResultType = typename TestFixture::type;

    constexpr size_t size = 1313;
    ResultType*      data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(ResultType) * size));

    rocrand_scrambled_sobol32 g;
    ROCRAND_CHECK(g.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<ResultType> host_data(size);
    HIP_CHECK(hipMemcpy(host_data.data(), data, sizeof(ResultType) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));

    double mean = 0.0;
    for(ResultType v : host_data)
    {
        mean += v;
    }
    mean = mean / static_cast<double>(std::numeric_limits<ResultType>::max());
    mean = mean / size;

    ASSERT_NEAR(mean, 0.5, 0.05);
}

TYPED_TEST(rocrand_scrambled_sobol32_float_tests, normal_test)
{
    using ResultType = typename TestFixture::type;

    ResultType ExpectedMean = 2.0f;
    ResultType ExpectedStd  = 5.0f;

    constexpr size_t size = 1313;
    ResultType*      data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(ResultType) * size));

    rocrand_scrambled_sobol32 g;
    ROCRAND_CHECK(g.generate_normal(data, size, ExpectedMean, ExpectedStd));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<ResultType> host_data(size);
    HIP_CHECK(hipMemcpy(host_data.data(), data, sizeof(ResultType) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));

    double mean = 0.0;
    double std  = 0.0;
    for(ResultType v : host_data)
    {
        mean += v;
    }
    mean = mean / size;
    for(ResultType v : host_data)
    {
        std += std::pow(v - mean, 2);
    }
    std = sqrt(std / size);

    EXPECT_NEAR(ExpectedMean, mean, ExpectedMean * 0.1); // 10%
    EXPECT_NEAR(ExpectedStd, std, ExpectedStd * 0.1); // 10%
}

TEST(rocrand_scrambled_sobol32_qrng_tests, poisson_test)
{
    constexpr size_t size = 1313;
    unsigned int*    data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

    rocrand_scrambled_sobol32 g;
    ROCRAND_CHECK(g.generate_poisson(data, size, 5.5));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> host_data(size);
    HIP_CHECK(
        hipMemcpy(host_data.data(), data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));

    double mean = 0.0;
    double var  = 0.0;
    for(unsigned int v : host_data)
    {
        mean += v;
    }
    mean = mean / size;

    for(size_t i = 0; i < size; i++)
    {
        double x = host_data[i] - mean;
        var += x * x;
    }
    var = var / size;

    EXPECT_NEAR(mean, 5.5, std::max(1.0, 5.5 * 1e-2));
    EXPECT_NEAR(var, 5.5, std::max(1.0, 5.5 * 1e-2));
}

TEST(rocrand_scrambled_sobol32_qrng_tests, dimensions_test)
{
    const size_t size = 12345;
    float*       data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(float) * size));

    rocrand_scrambled_sobol32 g;

    ROCRAND_CHECK(g.generate(data, size));

    ROCRAND_CHECK(g.set_dimensions(4));
    EXPECT_EQ(g.generate(data, size), ROCRAND_STATUS_LENGTH_NOT_MULTIPLE);

    ROCRAND_CHECK(g.set_dimensions(15));
    ROCRAND_CHECK(g.generate(data, size));

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(data));
}

// Check if the numbers generated by first generate() call are different from
// the numbers generated by the 2nd call (same generator)
TEST(rocrand_scrambled_sobol32_qrng_tests, state_progress_test)
{
    // Device data
    constexpr size_t size = 1025;
    unsigned int*    data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

    // Generator
    rocrand_scrambled_sobol32 g0;

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> host_data1(size);
    HIP_CHECK(
        hipMemcpy(host_data1.data(), data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> host_data2(size);
    HIP_CHECK(
        hipMemcpy(host_data2.data(), data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));

    size_t same = 0;
    for(size_t i = 0; i < size; i++)
    {
        if(host_data1[i] == host_data2[i])
            same++;
    }
    // It may happen that numbers are the same, so we
    // just make sure that most of them are different.
    EXPECT_LT(same, static_cast<size_t>(0.01f * size));
}

TEST(rocrand_scrambled_sobol32_qrng_tests, discard_test)
{
    const unsigned int* h_directions;
    const unsigned int* h_constants;

    ROCRAND_CHECK(rocrand_get_direction_vectors32(&h_directions,
                                                  ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
    ROCRAND_CHECK(rocrand_get_scramble_constants32(&h_constants));

    rocrand_scrambled_sobol32::engine_type engine1(&h_directions[32], h_constants[1], 678);
    rocrand_scrambled_sobol32::engine_type engine2(&h_directions[32], h_constants[1], 676);

    EXPECT_NE(engine1(), engine2());

    engine2.discard();

    EXPECT_NE(engine1(), engine2());

    engine2.discard();

    EXPECT_EQ(engine1(), engine2());
    EXPECT_EQ(engine1(), engine2());

    const unsigned int ds[] = {0, 1, 4, 37, 583, 7452, 21032, 35678, 66778, 10313475, 82120230};

    for(auto d : ds)
    {
        for(unsigned int i = 0; i < d; i++)
        {
            engine1.discard();
        }
        engine2.discard(d);

        EXPECT_EQ(engine1(), engine2());
    }
}

TEST(rocrand_scrambled_sobol32_qrng_tests, discard_stride_test)
{
    const unsigned int* h_directions;
    const unsigned int* h_constants;

    ROCRAND_CHECK(rocrand_get_direction_vectors32(&h_directions,
                                                  ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6));
    ROCRAND_CHECK(rocrand_get_scramble_constants32(&h_constants));

    rocrand_scrambled_sobol32::engine_type engine1(&h_directions[32], h_constants[1], 123);
    rocrand_scrambled_sobol32::engine_type engine2(&h_directions[32], h_constants[1], 123);

    EXPECT_EQ(engine1(), engine2());

    const unsigned int ds[] = {1, 10, 12, 20, 4, 5, 30};

    for(auto d : ds)
    {
        engine1.discard(1 << d);
        engine2.discard_stride(1 << d);

        EXPECT_EQ(engine1(), engine2());
    }
}

class rocrand_scrambled_sobol32_qrng_offset
    : public ::testing::TestWithParam<std::tuple<unsigned int, unsigned long long>>
{};

TEST_P(rocrand_scrambled_sobol32_qrng_offset, offsets_test)
{
    const unsigned int       dimensions = std::get<0>(GetParam());
    const unsigned long long offset     = std::get<1>(GetParam());

    const size_t size = 1313;

    const size_t  size0 = size * dimensions;
    const size_t  size1 = (size + offset) * dimensions;
    unsigned int* data0;
    unsigned int* data1;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data0), sizeof(unsigned int) * size0));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data1), sizeof(unsigned int) * size1));

    rocrand_scrambled_sobol32 g0;
    g0.set_offset(offset);
    g0.set_dimensions(dimensions);
    g0.generate(data0, size0);

    rocrand_scrambled_sobol32 g1;
    g1.set_dimensions(dimensions);
    g1.generate(data1, size1);

    std::vector<unsigned int> host_data0(size0);
    std::vector<unsigned int> host_data1(size1);
    HIP_CHECK(
        hipMemcpy(host_data0.data(), data0, sizeof(unsigned int) * size0, hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(host_data1.data(), data1, sizeof(unsigned int) * size1, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    for(unsigned int d = 0; d < dimensions; d++)
    {
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(host_data0[d * size + i], host_data1[d * (size + offset) + i + offset]);
        }
    }

    HIP_CHECK(hipFree(data0));
    HIP_CHECK(hipFree(data1));
}

const unsigned int       dimensions[] = {1, 2, 10, 321};
const unsigned long long offsets[]    = {0, 1, 11, 112233};

INSTANTIATE_TEST_SUITE_P(rocrand_scrambled_sobol32_qrng_offset,
                         rocrand_scrambled_sobol32_qrng_offset,
                         ::testing::Combine(::testing::ValuesIn(dimensions),
                                            ::testing::ValuesIn(offsets)));

class rocrand_scrambled_sobol32_qrng_continuity : public ::testing::TestWithParam<unsigned int>
{};

// Check that subsequent generations of different sizes produce one Sobol
// sequence without gaps, no matter how many values are generated per call.
TEST_P(rocrand_scrambled_sobol32_qrng_continuity, continuity_test)
{
    const unsigned int dimensions = GetParam();

    const std::vector<size_t> sizes0({100, 1, 24783, 3, 2, 776543});
    const std::vector<size_t> sizes1({1024, 56, 65536, 623456, 30, 111330});

    const size_t s0 = std::accumulate(sizes0.cbegin(), sizes0.cend(), std::size_t{0});
    const size_t s1 = std::accumulate(sizes1.cbegin(), sizes1.cend(), std::size_t{0});

    const size_t size0 = s0 * dimensions;
    const size_t size1 = s1 * dimensions;

    unsigned int* data0;
    unsigned int* data1;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data0), sizeof(unsigned int) * size0));
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data1), sizeof(unsigned int) * size1));

    rocrand_scrambled_sobol32 g0;
    rocrand_scrambled_sobol32 g1;
    g0.set_dimensions(dimensions);
    g1.set_dimensions(dimensions);

    std::vector<unsigned int> host_data0(size0);
    std::vector<unsigned int> host_data1(size1);

    // host_data0 contains all s0 values of dim0, then all s0 values of dim1...
    // host_data1 contains all s1 values of dim0, then all s1 values of dim1...
    size_t current0 = 0;
    for(size_t s : sizes0)
    {
        g0.generate(data0, s * dimensions);
        for(unsigned int d = 0; d < dimensions; d++)
        {
            HIP_CHECK(hipMemcpy(host_data0.data() + s0 * d + current0,
                                data0 + d * s,
                                sizeof(unsigned int) * s,
                                hipMemcpyDefault));
        }
        current0 += s;
    }
    size_t current1 = 0;
    for(size_t s : sizes1)
    {
        g1.generate(data1, s * dimensions);
        for(unsigned int d = 0; d < dimensions; d++)
        {
            HIP_CHECK(hipMemcpy(host_data1.data() + s1 * d + current1,
                                data1 + d * s,
                                sizeof(unsigned int) * s,
                                hipMemcpyDefault));
        }
        current1 += s;
    }

    for(unsigned int d = 0; d < dimensions; d++)
    {
        for(size_t i = 0; i < std::min(s0, s1); i++)
        {
            ASSERT_EQ(host_data0[d * s0 + i], host_data1[d * s1 + i]);
        }
    }

    HIP_CHECK(hipFree(data0));
    HIP_CHECK(hipFree(data1));
}

const unsigned int continuity_test_dimensions[] = {1, 2, 10, 21};

INSTANTIATE_TEST_SUITE_P(rocrand_scrambled_sobol32_qrng_continuity,
                         rocrand_scrambled_sobol32_qrng_continuity,
                         ::testing::ValuesIn(continuity_test_dimensions));
