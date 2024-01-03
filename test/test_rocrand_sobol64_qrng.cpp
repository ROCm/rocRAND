// Copyright (c) 2021-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include "test_utils_hipgraphs.hpp"

#include <rng/sobol64.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)
#define ROCRAND_CHECK(state) ASSERT_EQ(state, ROCRAND_STATUS_SUCCESS)

class rocrand_sobol64_qrng_tests : public ::testing::TestWithParam<bool> { };

#if 0
struct int128
{
    unsigned long long int data[2] = {0, 0};
    int128& operator+=(unsigned long long int operand)
    {
        if(std::abs<long long>(data[0] - operand) > (UINT64_MAX - data[0]))
        {
            data[1]++;
        }
        data[0] += operand;
        return *this;
    }
};

std::ostream &operator<<(std::ostream &output, const int128& ref)
{
    output << std::hex << "0x" << ref.data[1] << " : 0x" << ref.data[0];
    return output;
}

TEST_P(rocrand_sobol64_qrng_tests, uniform_uint64_test)
{
    const bool use_graphs = GetParam();
    using T = unsigned long long int;
    //constexpr size_t size = 1313;
    constexpr size_t size = 1313;
    T * data;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), sizeof(T) * size));

    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        rocrand_sobol64 g;
        ROCRAND_CHECK(rocrand_set_stream(&g, stream));
        ROCRAND_CHECK(g.generate(data, size));
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(T) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    //unsigned long long int sum = 0;
    int128 sum;
    for(size_t i = 0; i < size; i++)
    {
        //std::cout << host_data[i] << " - ";
        sum += host_data[i];
        //std::cout << sum << "\n";
    }
    //std::cout << std::endl;
    //const T mean = sum / size;
    //ASSERT_NEAR(mean, UINT64_MAX / 2lu, UINT64_MAX / 20lu);

    HIP_CHECK(hipFree(data));
    HIP_CHECK(hipFree(data));
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}
#endif

TEST_P(rocrand_sobol64_qrng_tests, uniform_double_test)
{
    const bool use_graphs = GetParam();
    constexpr size_t size = 1313;
    double * data;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), sizeof(double) * size));

    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        rocrand_sobol64 g;
        ROCRAND_CHECK(rocrand_set_stream(&g, stream));
        ROCRAND_CHECK(g.generate(data, size));
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());

    double host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(double) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double sum = 0;
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_GT(host_data[i], 0.0f);
        ASSERT_LE(host_data[i], 1.0f);
        sum += host_data[i];
    }
    const double mean = sum / size;
    ASSERT_NEAR(mean, 0.5f, 0.05f);

    HIP_CHECK(hipFree(data));
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST_P(rocrand_sobol64_qrng_tests, uniform_uint_test)
{
    const bool use_graphs = GetParam();
    const size_t size = 1313;
    unsigned int * data;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        rocrand_sobol64 g;
        ROCRAND_CHECK(rocrand_set_stream(&g, stream));
        ROCRAND_CHECK(g.generate(data, size));
    }
    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long int sum = 0;
    for(size_t i = 0; i < size; i++)
    {
        sum += host_data[i];
    }

    const unsigned int mean = sum / size;
    ASSERT_NEAR(mean, UINT_MAX / 2, UINT_MAX / 20);

    HIP_CHECK(hipFree(data));
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST_P(rocrand_sobol64_qrng_tests, normal_double_test)
{
    const bool use_graphs = GetParam();
    const size_t size = 1313;
    double * data;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), sizeof(double) * size));

    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        rocrand_sobol64 g;
        ROCRAND_CHECK(rocrand_set_stream(&g, stream));
        ROCRAND_CHECK(g.generate_normal(data, size, 2.0, 5.0));
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());

    double host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(double) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;

    double std = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        std += std::pow(host_data[i] - mean, 2);
    }
    std = sqrt(std / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, std, 1.0); // 20%

    HIP_CHECK(hipFree(data));
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST_P(rocrand_sobol64_qrng_tests, poisson_test)
{
    const bool use_graphs = GetParam();
    using T = unsigned int;
    constexpr size_t size = 1313;
    T * data;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), sizeof(T) * size));

    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        rocrand_sobol64 g;
        ROCRAND_CHECK(rocrand_set_stream(&g, stream));
        ROCRAND_CHECK(g.generate_poisson(data, size, 5.5));
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(T) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;

    double var = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        double x = host_data[i] - mean;
        var += x * x;
    }
    var = var / size;

    EXPECT_NEAR(mean, 5.5, std::max(1.0, 5.5 * 1e-2));
    EXPECT_NEAR(var, 5.5, std::max(1.0, 5.5 * 1e-2));

    HIP_CHECK(hipFree(data));
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST_P(rocrand_sobol64_qrng_tests, dimesions_test)
{
    const bool use_graphs = GetParam();
    const size_t size = 12345;
    double * data;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), sizeof(double) * size));

    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        rocrand_sobol64 g;
        ROCRAND_CHECK(rocrand_set_stream(&g, stream));
        ROCRAND_CHECK(g.generate(data, size));

        ROCRAND_CHECK(g.set_dimensions(4));
        EXPECT_EQ(g.generate(data, size), ROCRAND_STATUS_LENGTH_NOT_MULTIPLE);

        ROCRAND_CHECK(g.set_dimensions(15));
        ROCRAND_CHECK(g.generate(data, size));
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

// Check if the numbers generated by first generate() call are different from
// the numbers generated by the 2nd call (same generator)
TEST_P(rocrand_sobol64_qrng_tests, state_progress_test)
{
    const bool use_graphs = GetParam();

    // Device data
    const size_t size = 1025;
    unsigned int * data;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    unsigned int host_data1[size];
    unsigned int host_data2[size];

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        // Generator
        rocrand_sobol64 g0;
        ROCRAND_CHECK(rocrand_set_stream(&g0, stream));

        // Generate using g0 and copy to host
        ROCRAND_CHECK(g0.generate(data, size));
        if (use_graphs)
            graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
        else
            HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemcpy(host_data1, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        if (use_graphs)
            test_utils::resetGraphHelper(graph, graph_instance, stream);

        // Generate using g0 and copy to host
        ROCRAND_CHECK(g0.generate(data, size));
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());
    
    HIP_CHECK(hipMemcpy(host_data2, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    size_t same = 0;
    for(size_t i = 0; i < size; i++)
    {
        if(host_data1[i] == host_data2[i]) same++;
    }
    // It may happen that numbers are the same, so we
    // just make sure that most of them are different.
    EXPECT_LT(same, static_cast<size_t>(0.01f * size));

    HIP_CHECK(hipFree(data));
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

INSTANTIATE_TEST_SUITE_P(rocrand_sobol64_qrng_tests,
                         rocrand_sobol64_qrng_tests,
                         ::testing::Bool());

TEST(rocrand_sobol64_qrng_engine_tests, discard_test)
{
    const unsigned long long int* h_directions;
    rocrand_get_direction_vectors64(&h_directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6);

    rocrand_sobol64::engine_type engine1(&h_directions[32], 678ll);
    rocrand_sobol64::engine_type engine2(&h_directions[32], 676ll);

    EXPECT_NE(engine1(), engine2());

    engine2.discard();

    EXPECT_NE(engine1(), engine2());

    engine2.discard();

    EXPECT_EQ(engine1(), engine2());
    EXPECT_EQ(engine1(), engine2());

    const unsigned int ds[] = {
        0, 1, 4, 37, 583, 7452,
        21032, 35678, 66778, 10313475, 82120230
    };

    for (auto d : ds)
    {
        for (unsigned int i = 0; i < d; i++)
        {
            engine1.discard();
        }
        engine2.discard(d);

        EXPECT_EQ(engine1(), engine2());
    }
}

TEST(rocrand_sobol64_qrng_engine_tests, discard_stride_test)
{
    const unsigned long long int* h_directions;
    rocrand_get_direction_vectors64(&h_directions, ROCRAND_DIRECTION_VECTORS_64_JOEKUO6);

    rocrand_sobol64::engine_type engine1(&h_directions[64], 123);
    rocrand_sobol64::engine_type engine2(&h_directions[64], 123);

    EXPECT_EQ(engine1(), engine2());

    const unsigned int ds[] = {
        1, 10, 12, 20, 4, 5, 30
    };

    for (auto d : ds)
    {
        engine1.discard(1 << d);
        engine2.discard_stride(1 << d);

        EXPECT_EQ(engine1(), engine2());
    }
}

class rocrand_sobol64_qrng_offset
    : public ::testing::TestWithParam<std::tuple<unsigned int, unsigned long long, bool>> { };

TEST_P(rocrand_sobol64_qrng_offset, offsets_test)
{
    const unsigned int dimensions   = std::get<0>(GetParam());
    const unsigned long long offset = std::get<1>(GetParam());
    const bool use_graphs           = std::get<2>(GetParam());

    const size_t size = 1313;

    const size_t size0 = size * dimensions;
    const size_t size1 = (size + offset) * dimensions;
    unsigned int * data0;
    unsigned int * data1;
    hipMalloc(reinterpret_cast<void**>(&data0), sizeof(unsigned int) * size0);
    hipMalloc(reinterpret_cast<void**>(&data1), sizeof(unsigned int) * size1);

    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        rocrand_sobol64 g0;
        ROCRAND_CHECK(rocrand_set_stream(&g0, stream));

        g0.set_offset(offset);
        g0.set_dimensions(dimensions);
        g0.generate(data0, size0);

        rocrand_sobol64 g1;
        ROCRAND_CHECK(rocrand_set_stream(&g1, stream));
        g1.set_dimensions(dimensions);
        g1.generate(data1, size1);
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        hipDeviceSynchronize();

    std::vector<unsigned int> host_data0(size0);
    std::vector<unsigned int> host_data1(size1);
    hipMemcpy(host_data0.data(), data0, sizeof(unsigned int) * size0, hipMemcpyDeviceToHost);
    hipMemcpy(host_data1.data(), data1, sizeof(unsigned int) * size1, hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    for(unsigned int d = 0; d < dimensions; d++)
    {
        for(size_t i = 0; i < size; i++)
        {
            ASSERT_EQ(
                host_data0[d * size + i],
                host_data1[d * (size + offset) + i + offset]
            );
        }
    }

    hipFree(data0);
    hipFree(data1);
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

const unsigned int dimensions[] = { 1, 2, 10, 321 };
const unsigned long long offsets[] = { 0, 1, 11, 112233 };

INSTANTIATE_TEST_SUITE_P(rocrand_sobol64_qrng_offset,
                        rocrand_sobol64_qrng_offset,
                        ::testing::Combine(
                            ::testing::ValuesIn(dimensions), 
                            ::testing::ValuesIn(offsets),
                            ::testing::Bool()));

class rocrand_sobol64_qrng_continuity
    : public ::testing::TestWithParam<std::tuple<unsigned int, bool>> { };

// Check that subsequent generations of different sizes produce one Sobol
// sequence without gaps, no matter how many values are generated per call.
TEST_P(rocrand_sobol64_qrng_continuity, continuity_test)
{
    const unsigned int dimensions = std::get<0>(GetParam());
    const bool use_graphs         = std::get<1>(GetParam());

    const std::vector<size_t> sizes0({ 100, 1, 24783, 3, 2, 776543 });
    const std::vector<size_t> sizes1({ 1024, 56, 65536, 623456, 30, 111330 });

    const size_t s0 = std::accumulate(sizes0.cbegin(), sizes0.cend(), std::size_t{0});
    const size_t s1 = std::accumulate(sizes1.cbegin(), sizes1.cend(), std::size_t{0});

    const size_t size0 = s0 * dimensions;
    const size_t size1 = s1 * dimensions;

    unsigned int * data0;
    unsigned int * data1;
    hipMalloc(reinterpret_cast<void**>(&data0), sizeof(unsigned int) * size0);
    hipMalloc(reinterpret_cast<void**>(&data1), sizeof(unsigned int) * size1);

    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    std::vector<unsigned int> host_data0(size0);
    std::vector<unsigned int> host_data1(size1);

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        rocrand_sobol64 g0;
        ROCRAND_CHECK(rocrand_set_stream(&g0, stream));
        rocrand_sobol64 g1;
        ROCRAND_CHECK(rocrand_set_stream(&g1, stream));

        g0.set_dimensions(dimensions);
        g1.set_dimensions(dimensions);

        // host_data0 contains all s0 values of dim0, then all s0 values of dim1...
        // host_data1 contains all s1 values of dim0, then all s1 values of dim1...
        size_t current0 = 0;
        for (size_t s : sizes0)
        {
            g0.generate(data0, s * dimensions);
            for(unsigned int d = 0; d < dimensions; d++)
            {
                hipMemcpyAsync(
                    host_data0.data() + s0 * d + current0,
                    data0 + d * s,
                    sizeof(unsigned int) * s, hipMemcpyDefault,
                    stream);
            }
            current0 += s;
        }
        size_t current1 = 0;
        for (size_t s : sizes1)
        {
            g1.generate(data1, s * dimensions);
            for(unsigned int d = 0; d < dimensions; d++)
            {
                hipMemcpyAsync(
                    host_data1.data() + s1 * d + current1,
                    data1 + d * s,
                    sizeof(unsigned int) * s, hipMemcpyDefault,
                    stream);
            }
            current1 += s;
        }
    }
    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);

    for(unsigned int d = 0; d < dimensions; d++)
    {
        for(size_t i = 0; i < std::min(s0, s1); i++)
        {
            ASSERT_EQ(
                host_data0[d * s0 + i],
                host_data1[d * s1 + i]
            );
        }
    }

    hipFree(data0);
    hipFree(data1);
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

const unsigned int continuity_test_dimensions[] = { 1, 2, 10, 21 };

INSTANTIATE_TEST_SUITE_P(rocrand_sobol64_qrng_continuity,
                         rocrand_sobol64_qrng_continuity,
                         ::testing::Combine(
                            ::testing::ValuesIn(continuity_test_dimensions),
                            ::testing::Bool()));
