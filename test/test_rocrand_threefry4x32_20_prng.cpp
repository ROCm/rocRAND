// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <numeric>
#include <stdio.h>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include <rng/generator_type.hpp>
#include <rng/generators.hpp>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include "test_utils_hipgraphs.hpp"

class rocrand_threefry_prng_tests : public ::testing::TestWithParam<bool> { };

// Assert that the kernel arguments are trivially copyable and destructible.
TEST(rocrand_threefry_prng_tests, type)
{
    typedef ::rocrand_host::detail::threefry4x32_20_device_engine engine_type;
    // TODO: Enable once uint4 is trivially copyable.
    // EXPECT_TRUE(std::is_trivially_copyable<engine_type>::value);
    EXPECT_TRUE(std::is_trivially_destructible<engine_type>::value);
}

TEST_P(rocrand_threefry_prng_tests, uniform_uint_test)
{
    const bool use_graphs = GetParam();
    const size_t  size = 1313;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * (size + 1)));
    
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

        rocrand_threefry4x32_20 g;
        ROCRAND_CHECK(rocrand_set_stream(&g, stream));
        ROCRAND_CHECK(g.generate(data + 1, size));
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data + 1, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long sum = 0;
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

TEST_P(rocrand_threefry_prng_tests, uniform_float_test)
{
    const bool use_graphs = GetParam();
    const size_t size = 1313;
    float*       data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(float) * size));

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

        rocrand_threefry4x32_20 g;
        ROCRAND_CHECK(rocrand_set_stream(&g, stream));
        ROCRAND_CHECK(g.generate(data, size));
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());

    float host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(float) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double sum = 0;
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_GT(host_data[i], 0.0f);
        ASSERT_LE(host_data[i], 1.0f);
        sum += host_data[i];
    }
    const float mean = sum / size;
    ASSERT_NEAR(mean, 0.5f, 0.05f);

    HIP_CHECK(hipFree(data));
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

// Check if the numbers generated by first generate() call are different from
// the numbers generated by the 2nd call (same generator)
TEST_P(rocrand_threefry_prng_tests, state_progress_test)
{
    const bool use_graphs = GetParam();
    // Device data
    const size_t  size = 1025;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));
    
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
        rocrand_threefry4x32_20 g0;
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
        if(host_data1[i] == host_data2[i])
            same++;
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

// Checks if generators with the same seed and in the same state
// generate the same numbers
TEST_P(rocrand_threefry_prng_tests, same_seed_test)
{
    const bool use_graphs = GetParam();
    const unsigned long long seed = 0xdeadbeefdeadbeefULL;

    // Device side data
    const size_t  size = 1024;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));
    
    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    unsigned int g0_host_data[size];
    unsigned int g1_host_data[size];

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        // Generators
        rocrand_threefry4x32_20 g0;
        ROCRAND_CHECK(rocrand_set_stream(&g0, stream));
        rocrand_threefry4x32_20 g1;
        ROCRAND_CHECK(rocrand_set_stream(&g1, stream));

        // Set same seeds
        g0.set_seed(seed);
        g1.set_seed(seed);

        // Generate using g0 and copy to host
        ROCRAND_CHECK(g0.generate(data, size));

        if (use_graphs)
            graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
        else
            HIP_CHECK(hipDeviceSynchronize());

        
        HIP_CHECK(hipMemcpy(g0_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        if (use_graphs)
            test_utils::resetGraphHelper(graph, graph_instance, stream);

        // Generate using g1 and copy to host
        ROCRAND_CHECK(g1.generate(data, size));
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(g1_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Numbers generated using same generator with same
    // seed should be the same
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(g0_host_data[i], g1_host_data[i]);
    }

    HIP_CHECK(hipFree(data));
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

// Checks if generators with the same seed and in the same state generate
// the same numbers
TEST_P(rocrand_threefry_prng_tests, different_seed_test)
{
    const bool use_graphs = GetParam();
    const unsigned long long seed0 = 0xdeadbeefdeadbeefULL;
    const unsigned long long seed1 = 0xbeefdeadbeefdeadULL;

    // Device side data
    const size_t  size = 1024;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(reinterpret_cast<void**>(&data), sizeof(unsigned int) * size));

    hipStream_t stream = 0;
    hipGraph_t graph;
    hipGraphExec_t graph_instance;
    if (use_graphs)
    {
        // Default stream does not support hipGraph stream capture, so create a non-blocking one
        HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }

    unsigned int g0_host_data[size];
    unsigned int g1_host_data[size];

    // This anonymous block ensures that:
    // - generators are destroyed before the stream because their destructors may call hipFreeAsync
    // - the call to the generator destructor is captured inside the graph
    {
        if (use_graphs)
            graph = test_utils::createGraphHelper(stream);

        // Generators
        rocrand_threefry4x32_20 g0;
        ROCRAND_CHECK(rocrand_set_stream(&g0, stream));
        rocrand_threefry4x32_20 g1;
        ROCRAND_CHECK(rocrand_set_stream(&g1, stream));

        // Set different seeds
        g0.set_seed(seed0);
        g1.set_seed(seed1);
        ASSERT_NE(g0.get_seed(), g1.get_seed());

        // Generate using g0 and copy to host
        ROCRAND_CHECK(g0.generate(data, size));

        if (use_graphs)
            graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
        else
            HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemcpy(g0_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        if (use_graphs)
            test_utils::resetGraphHelper(graph, graph_instance, stream);

        // Generate using g1 and copy to host
        ROCRAND_CHECK(g1.generate(data, size));
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    else
        HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(g1_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    size_t same = 0;
    for(size_t i = 0; i < size; i++)
    {
        if(g1_host_data[i] == g0_host_data[i])
            same++;
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

INSTANTIATE_TEST_SUITE_P(rocrand_threefry_prng_tests,
                        rocrand_threefry_prng_tests,
                        testing::Bool()
);

///
/// rocrand_threefry_prng_state_tests TEST GROUP
///

// Just get access to internal state
class rocrand_threefry4x32_engine_type_test : public rocrand_threefry4x32_20::engine_type
{
public:
    __host__ rocrand_threefry4x32_engine_type_test() : rocrand_threefry4x32_20::engine_type(0, 0, 0)
    {}

    __host__ state_type& internal_state_ref()
    {
        return m_state;
    }
};

TEST(rocrand_threefry_prng_state_tests, seed_test)
{
    rocrand_threefry4x32_engine_type_test              engine;
    rocrand_threefry4x32_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 1U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.seed(3331, 0, 5 * 4ULL);
    EXPECT_EQ(state.counter.x, 5U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);
}

// Check if the threefry state counter is calculated correctly during
// random number generation.
TEST(rocrand_threefry_prng_state_tests, discard_test)
{
    rocrand_threefry4x32_engine_type_test              engine;
    rocrand_threefry4x32_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard(UINT_MAX * 4ULL);
    EXPECT_EQ(state.counter.x, UINT_MAX);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard(UINT_MAX * 4ULL);
    EXPECT_EQ(state.counter.x, UINT_MAX - 1);
    EXPECT_EQ(state.counter.y, 1U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard(2 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 2U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = UINT_MAX;
    state.counter.y = UINT_MAX;
    state.counter.z = UINT_MAX;
    state.counter.w = 0;
    state.substate  = 0;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 1U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = UINT_MAX;
    state.counter.y = UINT_MAX;
    state.counter.z = UINT_MAX;
    state.counter.w = 1;
    state.substate  = 0;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 2U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 123;
    state.counter.y = 456;
    state.counter.z = 789;
    state.counter.w = 999;
    state.substate  = 0;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 124U);
    EXPECT_EQ(state.counter.y, 456U);
    EXPECT_EQ(state.counter.z, 789U);
    EXPECT_EQ(state.counter.w, 999U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 123;
    state.counter.y = 0;
    state.counter.z = 0;
    state.counter.w = 0;
    state.substate  = 0;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 124U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = UINT_MAX - 1;
    state.counter.y = 2;
    state.counter.z = 3;
    state.counter.w = 4;
    state.substate  = 0;
    engine.discard(((1ull << 32) + 2ull) * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 4U);
    EXPECT_EQ(state.counter.z, 3U);
    EXPECT_EQ(state.counter.w, 4U);
    EXPECT_EQ(state.substate, 0U);
}

TEST(rocrand_threefry_prng_state_tests, discard_sequence_test)
{
    rocrand_threefry4x32_engine_type_test              engine;
    rocrand_threefry4x32_engine_type_test::state_type& state = engine.internal_state_ref();

    engine.discard_subsequence(UINT_MAX);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, UINT_MAX);
    EXPECT_EQ(state.counter.w, 0U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard_subsequence(UINT_MAX);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, UINT_MAX - 1);
    EXPECT_EQ(state.counter.w, 1U);
    EXPECT_EQ(state.substate, 0U);

    engine.discard_subsequence(2);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 2U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 123;
    state.counter.y = 456;
    state.counter.z = 789;
    state.counter.w = 999;
    state.substate  = 0;
    engine.discard_subsequence(1);
    EXPECT_EQ(state.counter.x, 123U);
    EXPECT_EQ(state.counter.y, 456U);
    EXPECT_EQ(state.counter.z, 790U);
    EXPECT_EQ(state.counter.w, 999U);
    EXPECT_EQ(state.substate, 0U);

    state.counter.x = 1;
    state.counter.y = 2;
    state.counter.z = UINT_MAX - 1;
    state.counter.w = 4;
    state.substate  = 0;
    engine.discard_subsequence((1ull << 32) + 2ull);
    EXPECT_EQ(state.counter.x, 1U);
    EXPECT_EQ(state.counter.y, 2U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 6U);
    EXPECT_EQ(state.substate, 0U);
}

template<typename T, bool UseGraphs = false>
struct ThreefryPrngOffsetParams
{
    using output_type = T;
    static constexpr bool use_graphs = UseGraphs;
};

template<class Params>
class rocrand_threefry_prng_offset : public ::testing::Test
{
public:
    using output_type = typename Params::output_type;
    bool use_graphs = Params::use_graphs;
};

using RocrandThreefryPrngOffsetTypes = ::testing::Types<
    ThreefryPrngOffsetParams<unsigned int>,
    ThreefryPrngOffsetParams<float>,
    ThreefryPrngOffsetParams<unsigned int, true>,
    ThreefryPrngOffsetParams<float, true>>;
    
TYPED_TEST_SUITE(rocrand_threefry_prng_offset, RocrandThreefryPrngOffsetTypes);

TYPED_TEST(rocrand_threefry_prng_offset, offsets_test)
{
    using T           = typename TestFixture::output_type;
    const size_t size = 131313;

    constexpr size_t offsets[] = {0, 1, 4, 11, 65536, 112233};

    for(const auto offset : offsets)
    {
        SCOPED_TRACE(::testing::Message() << "with offset=" << offset);

        const size_t size0 = size;
        const size_t size1 = (size + offset);
        T*           data0;
        T*           data1;
        hipMalloc(reinterpret_cast<void**>(&data0), sizeof(T) * size0);
        hipMalloc(reinterpret_cast<void**>(&data1), sizeof(T) * size1);

        hipStream_t stream = 0;
        hipGraph_t graph;
        hipGraphExec_t graph_instance;
        if (TestFixture::use_graphs)
        {
            // Default stream does not support hipGraph stream capture, so create a non-blocking one
            HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        }

        // This anonymous block ensures that:
        // - generators are destroyed before the stream because their destructors may call hipFreeAsync
        // - the call to the generator destructor is captured inside the graph
        {
            if (TestFixture::use_graphs)
                graph = test_utils::createGraphHelper(stream);

            rocrand_threefry4x32_20 g0;
            ROCRAND_CHECK(rocrand_set_stream(&g0, stream));

            g0.set_offset(offset);
            g0.generate(data0, size0);

            if (TestFixture::use_graphs)
                graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);

            rocrand_threefry4x32_20 g1;
            ROCRAND_CHECK(rocrand_set_stream(&g1, stream));

            if (TestFixture::use_graphs)
                test_utils::resetGraphHelper(graph, graph_instance, stream);

            g1.generate(data1, size1);
        }

        if (TestFixture::use_graphs)
            graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);

        std::vector<T> host_data0(size0);
        std::vector<T> host_data1(size1);
        hipMemcpy(host_data0.data(), data0, sizeof(T) * size0, hipMemcpyDeviceToHost);
        hipMemcpy(host_data1.data(), data1, sizeof(T) * size1, hipMemcpyDeviceToHost);
        hipDeviceSynchronize();

        for(size_t i = 0; i < size; ++i)
        {
            ASSERT_EQ(host_data0[i], host_data1[i + offset]);
        }

        hipFree(data0);
        hipFree(data1);
        if (TestFixture::use_graphs)
        {
            test_utils::cleanupGraphHelper(graph, graph_instance);
            HIP_CHECK(hipStreamDestroy(stream));
        }
    }
}

// Check that subsequent generations of different sizes produce one
// sequence without gaps, no matter how many values are generated per call.
template<typename T, typename GenerateFunc>
void continuity_test(GenerateFunc generate_func, unsigned int divisor = 1, const bool use_graphs = false)
{
    std::vector<size_t> sizes0({100, 1, 24783, 3, 2, 776543, 1048576});
    std::vector<size_t> sizes1({1024, 55, 65536, 623456, 30, 1048576, 111331});
    if(divisor > 1)
    {
        for(size_t& s : sizes0)
            s = (s + divisor - 1) & ~static_cast<size_t>(divisor - 1);
        for(size_t& s : sizes1)
            s = (s + divisor - 1) & ~static_cast<size_t>(divisor - 1);
    }

    const size_t size0 = std::accumulate(sizes0.cbegin(), sizes0.cend(), std::size_t{0});
    const size_t size1 = std::accumulate(sizes1.cbegin(), sizes1.cend(), std::size_t{0});

    T* data0;
    T* data1;
    hipMalloc(reinterpret_cast<void**>(&data0), sizeof(T) * size0);
    hipMalloc(reinterpret_cast<void**>(&data1), sizeof(T) * size1);

    std::vector<T> host_data0(size0);
    std::vector<T> host_data1(size1);

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

        rocrand_threefry4x32_20 g0;
        ROCRAND_CHECK(rocrand_set_stream(&g0, stream));
        rocrand_threefry4x32_20 g1;
        ROCRAND_CHECK(rocrand_set_stream(&g1, stream));

        size_t current0 = 0;
        for(auto it = sizes0.begin(); it != sizes0.end(); it++)
        {
            const size_t cur_size = *it;
            generate_func(g0, data0, cur_size);

            hipMemcpyAsync(host_data0.data() + current0, data0, sizeof(T) * cur_size, hipMemcpyDefault, stream);
            current0 += cur_size;
        }
        size_t current1 = 0;
        for(size_t s : sizes1)
        {
            generate_func(g1, data1, s);

            hipMemcpyAsync(host_data1.data() + current1, data1, sizeof(T) * s, hipMemcpyDefault, stream);
            current1 += s;
        }
    }

    if (use_graphs)
        graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);

    for(size_t i = 0; i < std::min(size0, size1); i++)
    {
        ASSERT_EQ(host_data0[i], host_data1[i]);
    }

    hipFree(data0);
    hipFree(data1);
    if (use_graphs)
    {
        test_utils::cleanupGraphHelper(graph, graph_instance);
        HIP_CHECK(hipStreamDestroy(stream));
    }
}

TEST_P(rocrand_threefry_prng_tests, continuity_uniform_char_test)
{
    typedef unsigned char output_type;
    continuity_test<output_type>([](rocrand_threefry4x32_20& g, output_type* data, size_t s)
                                 { g.generate(data, s); },
                                 uniform_distribution<output_type, unsigned int>::output_width,
                                 GetParam());
}

TEST_P(rocrand_threefry_prng_tests, continuity_uniform_uint_test)
{
    typedef unsigned int output_type;
    continuity_test<output_type>([](rocrand_threefry4x32_20& g, output_type* data, size_t s)
                                 { g.generate(data, s); },
                                 uniform_distribution<output_type, unsigned int>::output_width,
                                 GetParam());
}

TEST_P(rocrand_threefry_prng_tests, continuity_uniform_float_test)
{
    typedef float output_type;
    continuity_test<output_type>([](rocrand_threefry4x32_20& g, output_type* data, size_t s)
                                 { g.generate(data, s); },
                                 uniform_distribution<output_type, unsigned int>::output_width,
                                 GetParam());
}

TEST_P(rocrand_threefry_prng_tests, continuity_uniform_double_test)
{
    typedef double output_type;
    continuity_test<output_type>([](rocrand_threefry4x32_20& g, output_type* data, size_t s)
                                 { g.generate(data, s); },
                                 uniform_distribution<output_type, unsigned int>::output_width,
                                 GetParam());
}

TEST_P(rocrand_threefry_prng_tests, continuity_normal_float_test)
{
    typedef float output_type;
    continuity_test<output_type>([](rocrand_threefry4x32_20& g, output_type* data, size_t s)
                                 { g.generate_normal(data, s, 0.f, 1.f); },
                                 normal_distribution<output_type, unsigned int>::output_width,
                                 GetParam());
}

TEST_P(rocrand_threefry_prng_tests, continuity_normal_double_test)
{
    typedef double output_type;
    continuity_test<output_type>([](rocrand_threefry4x32_20& g, output_type* data, size_t s)
                                 { g.generate_normal(data, s, 0., 1.); },
                                 normal_distribution<output_type, unsigned int>::output_width,
                                 GetParam());
}

TEST_P(rocrand_threefry_prng_tests, continuity_log_normal_float_test)
{
    typedef float output_type;
    continuity_test<output_type>([](rocrand_threefry4x32_20& g, output_type* data, size_t s)
                                 { g.generate_log_normal(data, s, 0.f, 1.f); },
                                 normal_distribution<output_type, unsigned int>::output_width,
                                 GetParam());
}

TEST_P(rocrand_threefry_prng_tests, continuity_log_normal_double_test)
{
    typedef double output_type;
    continuity_test<output_type>([](rocrand_threefry4x32_20& g, output_type* data, size_t s)
                                 { g.generate_log_normal(data, s, 0., 1.); },
                                 normal_distribution<output_type, unsigned int>::output_width,
                                 GetParam());
}

TEST_P(rocrand_threefry_prng_tests, continuity_poisson_test)
{
    typedef unsigned int output_type;
    continuity_test<output_type>([](rocrand_threefry4x32_20& g, output_type* data, size_t s)
                                 { g.generate_poisson(data, s, 100.); },
                                 1,
                                 GetParam());
}
