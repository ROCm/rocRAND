// Copyright (c) 2017-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include <rocrand/rocrand.h>

#include <rng/philox4x32_10.hpp>

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

template<class Params>
struct rocrand_philox_prng_tests : public testing::Test
{
    using generator_t                                 = typename Params::generator_t;
    static inline constexpr rocrand_ordering ordering = Params::ordering;

    auto get_generator() const
    {
        generator_t g;
        if(g.set_order(ordering) != ROCRAND_STATUS_SUCCESS)
        {
            throw std::runtime_error("Could not set ordering for generator");
        }
        return g;
    }
};

template<class Generator, rocrand_ordering Ordering>
struct rocrand_philox_prng_tests_params
{
    using generator_t                                 = Generator;
    static inline constexpr rocrand_ordering ordering = Ordering;
};

TYPED_TEST_SUITE_P(rocrand_philox_prng_tests);

TYPED_TEST_P(rocrand_philox_prng_tests, init_test)
{
    auto g = TestFixture::get_generator(); // offset = 0
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1337);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1048576);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1 << 24);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset(1 << 28);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());

    g.set_offset((1ULL << 36) + 1234567ULL);
    ROCRAND_CHECK(g.init());
    HIP_CHECK(hipDeviceSynchronize());
}

TYPED_TEST_P(rocrand_philox_prng_tests, uniform_uint_test)
{
    const size_t  size = 1313;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * (size + 1)));

    auto g = TestFixture::get_generator();
    ROCRAND_CHECK(g.generate_uniform(data + 1, size));
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
}

template<class Generator, class T>
void uniform_floating_point_test()
{
    const size_t size = 1313;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    Generator g;
    ROCRAND_CHECK(g.generate_uniform(data, size));

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(*host_data) * size, hipMemcpyDeviceToHost));

    double sum = 0;
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_GT(host_data[i], static_cast<T>(0.0));
        ASSERT_LE(host_data[i], static_cast<T>(1.0));
        sum += host_data[i];
    }
    const double mean = sum / size;
    ASSERT_NEAR(mean, 0.5, 0.05);

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(rocrand_philox_prng_tests, uniform_float_test)
{
    using generator_t = typename TestFixture::generator_t;

    uniform_floating_point_test<generator_t, float>();
}

TYPED_TEST_P(rocrand_philox_prng_tests, uniform_double_test)
{
    using generator_t = typename TestFixture::generator_t;

    uniform_floating_point_test<generator_t, double>();
}

template<class Generator, class T>
void normal_floating_point_test()
{
    const size_t size = 1313;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    Generator g;
    ROCRAND_CHECK(g.generate_normal(data, size, static_cast<T>(2.0), static_cast<T>(5.0)));
    HIP_CHECK(hipDeviceSynchronize());

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(*host_data) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;

    double stddev = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        stddev += std::pow(host_data[i] - mean, 2);
    }
    stddev = std::sqrt(stddev / size);

    EXPECT_NEAR(2.0, mean, 0.4); // 20%
    EXPECT_NEAR(5.0, stddev, 1.0); // 20%

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(rocrand_philox_prng_tests, normal_float_test)
{
    using generator_t = typename TestFixture::generator_t;

    normal_floating_point_test<generator_t, float>();
}

TYPED_TEST_P(rocrand_philox_prng_tests, normal_double_test)
{
    using generator_t = typename TestFixture::generator_t;

    normal_floating_point_test<generator_t, double>();
}

template<class Generator, class T>
void log_normal_floating_point_test()
{
    const size_t size = 131313;
    T*           data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(*data) * size));

    T normal_mean   = static_cast<T>(3.0);
    T normal_stddev = static_cast<T>(1.5);
    T normal_var    = normal_stddev * normal_stddev;

    T log_normal_mean   = std::exp(normal_mean + normal_var / 2.0);
    T log_normal_stddev = std::sqrt(std::exp(normal_var) - 1.0) * log_normal_mean;

    Generator g;
    ROCRAND_CHECK(g.generate_log_normal(data, size, normal_mean, normal_stddev));
    HIP_CHECK(hipDeviceSynchronize());

    T host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(*host_data) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    double mean = 0.0;
    for(size_t i = 0; i < size; i++)
    {
        mean += host_data[i];
    }
    mean = mean / size;

    double stddev = 0.0f;
    for(size_t i = 0; i < size; i++)
    {
        stddev += std::pow(host_data[i] - mean, 2);
    }
    stddev = std::sqrt(stddev / size);

    EXPECT_NEAR(log_normal_mean, mean, log_normal_mean * 0.2); // 20%
    EXPECT_NEAR(log_normal_stddev, stddev, log_normal_stddev * 0.2); // 20%

    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(rocrand_philox_prng_tests, log_normal_float_test)
{
    using generator_t = typename TestFixture::generator_t;

    log_normal_floating_point_test<generator_t, float>();
}

TYPED_TEST_P(rocrand_philox_prng_tests, log_normal_double_test)
{
    using generator_t = typename TestFixture::generator_t;

    log_normal_floating_point_test<generator_t, double>();
}

TYPED_TEST_P(rocrand_philox_prng_tests, poisson_test)
{
    const size_t  size = 1313;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    auto g = TestFixture::get_generator();
    ROCRAND_CHECK(g.generate_poisson(data, size, 5.5));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data[size];
    HIP_CHECK(hipMemcpy(host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
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
}

// Check if the numbers generated by first generate() call are different from
// the numbers generated by the 2nd call (same generator)
TYPED_TEST_P(rocrand_philox_prng_tests, state_progress_test)
{
    // Device data
    const size_t  size = 1025;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    // Generator
    auto g0 = TestFixture::get_generator();

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data1[size];
    HIP_CHECK(hipMemcpy(host_data1, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int host_data2[size];
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
}

// Checks if generators with the same seed and in the same state
// generate the same numbers
TYPED_TEST_P(rocrand_philox_prng_tests, same_seed_test)
{
    const unsigned long long seed = 0xdeadbeefdeadbeefULL;

    // Device side data
    const size_t  size = 1024;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    // Generators
    auto g0 = TestFixture::get_generator(), g1 = TestFixture::get_generator();
    // Set same seeds
    g0.set_seed(seed);
    g1.set_seed(seed);

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int g0_host_data[size];
    HIP_CHECK(hipMemcpy(g0_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Generate using g1 and copy to host
    ROCRAND_CHECK(g1.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int g1_host_data[size];
    HIP_CHECK(hipMemcpy(g1_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Numbers generated using same generator with same
    // seed should be the same
    for(size_t i = 0; i < size; i++)
    {
        ASSERT_EQ(g0_host_data[i], g1_host_data[i]);
    }
    HIP_CHECK(hipFree(data));
}

TYPED_TEST_P(rocrand_philox_prng_tests, different_seed_test)
{
    const unsigned long long seed0 = 0xdeadbeefdeadbeefULL;
    const unsigned long long seed1 = 0xbeefdeadbeefdeadULL;

    // Device side data
    const size_t  size = 1024;
    unsigned int* data;
    HIP_CHECK(hipMallocHelper(&data, sizeof(unsigned int) * size));

    // Generators
    auto g0 = TestFixture::get_generator(), g1 = TestFixture::get_generator();
    // Set different seeds
    g0.set_seed(seed0);
    g1.set_seed(seed1);
    ASSERT_NE(g0.get_seed(), g1.get_seed());

    // Generate using g0 and copy to host
    ROCRAND_CHECK(g0.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int g0_host_data[size];
    HIP_CHECK(hipMemcpy(g0_host_data, data, sizeof(unsigned int) * size, hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    // Generate using g1 and copy to host
    ROCRAND_CHECK(g1.generate(data, size));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int g1_host_data[size];
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
}

// Check that subsequent generations of different sizes produce one
// sequence without gaps, no matter how many values are generated per call.
template<typename T, typename Generator, typename GenerateFunc>
void continuity_test(GenerateFunc     generate_func,
                     rocrand_ordering ordering,
                     unsigned int     divisor = 1)
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
    HIP_CHECK(hipMalloc(&data0, sizeof(T) * size0));
    HIP_CHECK(hipMalloc(&data1, sizeof(T) * size1));

    Generator g0;
    g0.set_order(ordering);
    Generator g1;
    g1.set_order(ordering);

    std::vector<T> host_data0(size0);
    std::vector<T> host_data1(size1);

    size_t current0 = 0;
    for(size_t s : sizes0)
    {
        generate_func(g0, data0, s);
        HIP_CHECK(hipMemcpy(host_data0.data() + current0, data0, sizeof(T) * s, hipMemcpyDefault));
        current0 += s;
    }
    size_t current1 = 0;
    for(size_t s : sizes1)
    {
        generate_func(g1, data1, s);
        HIP_CHECK(hipMemcpy(host_data1.data() + current1, data1, sizeof(T) * s, hipMemcpyDefault));
        current1 += s;
    }

    for(size_t i = 0; i < std::min(size0, size1); i++)
    {
        ASSERT_EQ(host_data0[i], host_data1[i]);
    }

    HIP_CHECK(hipFree(data0));
    HIP_CHECK(hipFree(data1));
}

TYPED_TEST_P(rocrand_philox_prng_tests, continuity_uniform_uint_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef unsigned int output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_uniform(data, s); },
                                           ordering);
}

TYPED_TEST_P(rocrand_philox_prng_tests, continuity_uniform_char_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef unsigned char output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_uniform(data, s); },
                                           ordering,
                                           4);
}

TYPED_TEST_P(rocrand_philox_prng_tests, continuity_uniform_float_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef float output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_uniform(data, s); },
                                           ordering);
}

TYPED_TEST_P(rocrand_philox_prng_tests, continuity_uniform_double_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef double output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_uniform(data, s); },
                                           ordering);
}

TYPED_TEST_P(rocrand_philox_prng_tests, continuity_normal_float_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef float output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_normal(data, s, 0.0f, 1.0f); },
                                           ordering,
                                           2);
}

TYPED_TEST_P(rocrand_philox_prng_tests, continuity_normal_double_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef double output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_normal(data, s, 0.0, 1.0); },
                                           ordering,
                                           2);
}

TYPED_TEST_P(rocrand_philox_prng_tests, continuity_log_normal_float_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef float output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_log_normal(data, s, 0.0f, 1.0f); },
                                           ordering,
                                           2);
}

TYPED_TEST_P(rocrand_philox_prng_tests, continuity_log_normal_double_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef double output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_log_normal(data, s, 0.0, 1.0); },
                                           ordering,
                                           2);
}

TYPED_TEST_P(rocrand_philox_prng_tests, continuity_poisson_test)
{
    constexpr rocrand_ordering ordering = TestFixture::ordering;
    using generator_t                   = typename TestFixture::generator_t;
    typedef unsigned int output_t;

    continuity_test<output_t, generator_t>([](generator_t& g, output_t* data, size_t s)
                                           { g.generate_poisson(data, s, 100.0); },
                                           ordering);
}

template<class Params>
struct rocrand_philox_prng_offset_tests : public ::testing::Test
{
    using output_t                                    = typename Params::output_t;
    using generator_t                                 = typename Params::generator_t;
    static inline constexpr rocrand_ordering ordering = Params::ordering;

    auto get_generator() const
    {
        generator_t g;
        if(g.set_order(ordering) != ROCRAND_STATUS_SUCCESS)
        {
            throw std::runtime_error("Could not set ordering for generator");
        }
        return g;
    }
};

template<class Output, class Generator, rocrand_ordering Ordering>
struct rocrand_philox_prng_offset_tests_params
{
    using output_t                             = Output;
    using generator_t                          = Generator;
    static constexpr rocrand_ordering ordering = Ordering;
};

TYPED_TEST_SUITE_P(rocrand_philox_prng_offset_tests);

TYPED_TEST_P(rocrand_philox_prng_offset_tests, offsets_test)
{
    using output_t    = typename TestFixture::output_t;
    const size_t size = 131313;

    constexpr size_t offsets[] = {0, 1, 4, 11, 65536, 112233};

    for(const auto offset : offsets)
    {
        SCOPED_TRACE(::testing::Message() << "with offset=" << offset);

        const size_t size0 = size;
        const size_t size1 = (size + offset);
        output_t*    data0;
        output_t*    data1;
        HIP_CHECK(hipMalloc(&data0, sizeof(output_t) * size0));
        HIP_CHECK(hipMalloc(&data1, sizeof(output_t) * size1));

        auto g0 = TestFixture::get_generator();
        g0.set_offset(offset);
        g0.generate(data0, size0);

        auto g1 = TestFixture::get_generator();
        g1.generate(data1, size1);

        std::vector<output_t> host_data0(size0);
        std::vector<output_t> host_data1(size1);
        HIP_CHECK(
            hipMemcpy(host_data0.data(), data0, sizeof(output_t) * size0, hipMemcpyDeviceToHost));
        HIP_CHECK(
            hipMemcpy(host_data1.data(), data1, sizeof(output_t) * size1, hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        for(size_t i = 0; i < size; ++i)
        {
            ASSERT_EQ(host_data0[i], host_data1[i + offset]);
        }

        HIP_CHECK(hipFree(data0));
        HIP_CHECK(hipFree(data1));
    }
}

REGISTER_TYPED_TEST_SUITE_P(rocrand_philox_prng_tests,
                            init_test,
                            uniform_uint_test,
                            uniform_float_test,
                            uniform_double_test,
                            normal_float_test,
                            normal_double_test,
                            log_normal_float_test,
                            log_normal_double_test,
                            poisson_test,
                            state_progress_test,
                            same_seed_test,
                            different_seed_test,
                            continuity_uniform_uint_test,
                            continuity_uniform_char_test,
                            continuity_uniform_float_test,
                            continuity_uniform_double_test,
                            continuity_normal_float_test,
                            continuity_normal_double_test,
                            continuity_log_normal_float_test,
                            continuity_log_normal_double_test,
                            continuity_poisson_test);

REGISTER_TYPED_TEST_SUITE_P(rocrand_philox_prng_offset_tests, offsets_test);

// Generator API tests
using rocrand_philox_prng_tests_types = ::testing::Types<
    rocrand_philox_prng_tests_params<rocrand_philox4x32_10, ROCRAND_ORDERING_PSEUDO_DEFAULT>,
    rocrand_philox_prng_tests_params<rocrand_philox4x32_10, ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

using rocrand_philox_prng_offset_tests_types
    = ::testing::Types<rocrand_philox_prng_offset_tests_params<unsigned int,
                                                               rocrand_philox4x32_10,
                                                               ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       rocrand_philox_prng_offset_tests_params<unsigned int,
                                                               rocrand_philox4x32_10,
                                                               ROCRAND_ORDERING_PSEUDO_DYNAMIC>,
                       rocrand_philox_prng_offset_tests_params<float,
                                                               rocrand_philox4x32_10,
                                                               ROCRAND_ORDERING_PSEUDO_DEFAULT>,
                       rocrand_philox_prng_offset_tests_params<float,
                                                               rocrand_philox4x32_10,
                                                               ROCRAND_ORDERING_PSEUDO_DYNAMIC>>;

INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_philox_prng_tests,
                               rocrand_philox_prng_tests,
                               rocrand_philox_prng_tests_types);

INSTANTIATE_TYPED_TEST_SUITE_P(rocrand_philox_prng_offset_tests,
                               rocrand_philox_prng_offset_tests,
                               rocrand_philox_prng_offset_tests_types);

// Engine API tests
class rocrand_philox4x32_10_engine_type_test : public rocrand_philox4x32_10::engine_type
{
public:
    __host__ rocrand_philox4x32_10_engine_type_test() : rocrand_philox4x32_10::engine_type(0, 0, 0)
    {}

    __host__ state_type& internal_state_ref()
    {
        return m_state;
    }
};

TEST(rocrand_philox_prng_state_tests, seed_test)
{
    rocrand_philox4x32_10_engine_type_test              engine;
    rocrand_philox4x32_10_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 1U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    engine.seed(3331, 0, 5 * 4ULL);
    EXPECT_EQ(state.counter.x, 5U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);
}

// Check if the philox state counter is calculated correctly during
// random number generation.
TEST(rocrand_philox_prng_state_tests, discard_test)
{
    rocrand_philox4x32_10_engine_type_test              engine;
    rocrand_philox4x32_10_engine_type_test::state_type& state = engine.internal_state_ref();

    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    engine.discard(UINT_MAX * 4ULL);
    EXPECT_EQ(state.counter.x, UINT_MAX);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    engine.discard(UINT_MAX * 4ULL);
    EXPECT_EQ(state.counter.x, UINT_MAX - 1);
    EXPECT_EQ(state.counter.y, 1U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    engine.discard(2 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 2U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    state.counter.x = UINT_MAX;
    state.counter.y = UINT_MAX;
    state.counter.z = UINT_MAX;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 1U);

    state.counter.x = UINT_MAX;
    state.counter.y = UINT_MAX;
    state.counter.z = UINT_MAX;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 2U);

    state.counter.x = 123;
    state.counter.y = 456;
    state.counter.z = 789;
    state.counter.w = 999;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 124U);
    EXPECT_EQ(state.counter.y, 456U);
    EXPECT_EQ(state.counter.z, 789U);
    EXPECT_EQ(state.counter.w, 999U);

    state.counter.x = 123;
    state.counter.y = 0;
    state.counter.z = 0;
    state.counter.w = 0;
    engine.discard(1 * 4ULL);
    EXPECT_EQ(state.counter.x, 124U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 0U);

    state.counter.x = UINT_MAX - 1;
    state.counter.y = 2;
    state.counter.z = 3;
    state.counter.w = 4;
    engine.discard(((1ull << 32) + 2ull) * 4ULL);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 4U);
    EXPECT_EQ(state.counter.z, 3U);
    EXPECT_EQ(state.counter.w, 4U);
}

TEST(rocrand_philox_prng_state_tests, discard_sequence_test)
{
    rocrand_philox4x32_10_engine_type_test              engine;
    rocrand_philox4x32_10_engine_type_test::state_type& state = engine.internal_state_ref();

    engine.discard_subsequence(UINT_MAX);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, UINT_MAX);
    EXPECT_EQ(state.counter.w, 0U);

    engine.discard_subsequence(UINT_MAX);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, UINT_MAX - 1);
    EXPECT_EQ(state.counter.w, 1U);

    engine.discard_subsequence(2);
    EXPECT_EQ(state.counter.x, 0U);
    EXPECT_EQ(state.counter.y, 0U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 2U);

    state.counter.x = 123;
    state.counter.y = 456;
    state.counter.z = 789;
    state.counter.w = 999;
    engine.discard_subsequence(1);
    EXPECT_EQ(state.counter.x, 123U);
    EXPECT_EQ(state.counter.y, 456U);
    EXPECT_EQ(state.counter.z, 790U);
    EXPECT_EQ(state.counter.w, 999U);

    state.counter.x = 1;
    state.counter.y = 2;
    state.counter.z = UINT_MAX - 1;
    state.counter.w = 4;
    engine.discard_subsequence((1ull << 32) + 2ull);
    EXPECT_EQ(state.counter.x, 1U);
    EXPECT_EQ(state.counter.y, 2U);
    EXPECT_EQ(state.counter.z, 0U);
    EXPECT_EQ(state.counter.w, 6U);
}
