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

#include <vector>
#include <cmath>

#include <hip/hip_runtime.h>

#define QUALIFIERS __forceinline__ __host__ __device__
#include <hiprand_kernel.h>
#include <hiprand.h>

#define HIP_CHECK(x) ASSERT_EQ(x, hipSuccess)
#define HIPRAND_CHECK(state) ASSERT_EQ(state, HIPRAND_STATUS_SUCCESS)

template <class GeneratorState>
__global__
void hiprand_init_kernel(GeneratorState * states,
                         const size_t states_size,
                         unsigned long long seed,
                         unsigned long long offset)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int subsequence = state_id;
    if(state_id < states_size)
    {
        GeneratorState state;
        hiprand_init(seed, subsequence, offset, &state);
        states[state_id] = state;
    }
}

template <class GeneratorState>
__global__
void hiprand_skip_kernel(GeneratorState * states,
                         const size_t states_size,
                         unsigned long long seed,
                         unsigned long long offset)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int subsequence = state_id;
    if(state_id < states_size)
    {
        GeneratorState state;
        hiprand_init(seed, 0, offset, &state);
        skipahead(1ULL, &state);
        skipahead_sequence(subsequence, &state);
        states[state_id] = state;
    }
}

template <class GeneratorState>
__global__
void hiprand_kernel(unsigned int * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand(&state);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template <class GeneratorState>
__global__
void hiprand_uniform_kernel(float * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_uniform(&state);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template <class GeneratorState>
__global__
void hiprand_normal_kernel(float * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        float value;
        if(hipBlockIdx_x % 2 == 0)
            value = hiprand_normal2(&state).x;
        else
            value = hiprand_normal(&state);

        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template <class GeneratorState>
__global__
void hiprand_log_normal_kernel(float * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        float value;
        if(hipBlockIdx_x % 2 == 0)
            value = hiprand_log_normal2(&state, 1.6f, 0.25f).x;
        else
            value = hiprand_log_normal(&state, 1.6f, 0.25f);

        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template <class GeneratorState>
__global__
void hiprand_poisson_kernel(unsigned int * output, const size_t size, double lambda)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_poisson(&state, lambda);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template <class GeneratorState>
__global__
void hiprand_discrete_kernel(unsigned int * output, const size_t size, hiprandDiscreteDistribution_t discrete_distribution)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    GeneratorState state;
    const unsigned int subsequence = state_id;
    hiprand_init(12345, subsequence, 0, &state);

    unsigned int index = state_id;
    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = hiprand_discrete(&state, discrete_distribution);
        if(index < size)
            output[index] = value;
        index += global_size;
    }
}

template<class T>
void hiprand_kernel_h_hiprand_init_test()
{
    typedef T state_type;

    unsigned long long seed = 0xdeadbeefbeefdeadULL;
    unsigned long long offset = 4;

    const size_t states_size = 256;
    state_type * states;
    HIP_CHECK(hipMalloc((void **)&states, states_size * sizeof(state_type)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hiprand_init_kernel),
        dim3(4), dim3(64), 0, 0,
        states, states_size,
        seed, offset
    );
    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(states));
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand_init)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_init_test<state_type>();
}

TEST(hiprand_kernel_h_mrg32k3a, hiprand_init)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_init_test<state_type>();
}

TEST(hiprand_kernel_h_xorwow, hiprand_init)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_init_test<state_type>();
}

TEST(hiprand_kernel_h_default, hiprand_init)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_init_test<state_type>();
}

#ifdef __HIP_PLATFORM_NVCC__
TEST(hiprand_kernel_h_philox4x32_10, hiprand_init_nvcc)
{
    typedef hiprandStatePhilox4_32_10_t state_type;

    unsigned long long seed = 0xdeadbeefbeefdeadULL;
    unsigned long long offset = 4 * ((UINT_MAX * 17ULL) + 17);

    const size_t states_size = 256;
    state_type * states;
    HIP_CHECK(hipMalloc((void **)&states, states_size * sizeof(state_type)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hiprand_init_kernel),
        dim3(4), dim3(64), 0, 0,
        states, states_size,
        seed, offset
    );
    HIP_CHECK(hipPeekAtLastError());

    std::vector<state_type> states_host(states_size);
    HIP_CHECK(
        hipMemcpy(
            states_host.data(), states,
            states_size * sizeof(state_type),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(states));

    unsigned int subsequence = 0;
    for(auto& s : states_host)
    {
        EXPECT_EQ(s.key.x, 0xbeefdeadU);
        EXPECT_EQ(s.key.y, 0xdeadbeefU);

        EXPECT_EQ(s.ctr.x, 0U);
        EXPECT_EQ(s.ctr.y, 17U);
        EXPECT_EQ(s.ctr.z, subsequence);
        EXPECT_EQ(s.ctr.w, 0U);

        EXPECT_TRUE(
            s.output.x != 0U
            || s.output.y != 0U
            || s.output.z != 0U
            || s.output.w
        );

        EXPECT_EQ(s.STATE, 0U);

        subsequence++;
    }
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand_skip_nvcc)
{
    typedef hiprandStatePhilox4_32_10_t state_type;

    unsigned long long seed = 0xdeadbeefbeefdeadULL;
    unsigned long long offset = 4 * ((UINT_MAX * 17ULL) + 17);

    const size_t states_size = 256;
    state_type * states;
    HIP_CHECK(hipMalloc((void **)&states, states_size * sizeof(state_type)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hiprand_skip_kernel),
        dim3(4), dim3(64), 0, 0,
        states, states_size,
        seed, offset
    );
    HIP_CHECK(hipPeekAtLastError());

    std::vector<state_type> states_host(states_size);
    HIP_CHECK(
        hipMemcpy(
            states_host.data(), states,
            states_size * sizeof(state_type),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(states));

    unsigned int subsequence = 0;
    for(auto& s : states_host)
    {
        EXPECT_EQ(s.ctr.x, 0U);
        EXPECT_EQ(s.ctr.y, 17U);
        EXPECT_EQ(s.ctr.z, subsequence);
        EXPECT_EQ(s.ctr.w, 0U);
        EXPECT_EQ(s.STATE, 1U);
        subsequence++;
    }
}
#endif

template<class T>
void hiprand_kernel_h_hiprand_test()
{
    typedef T state_type;

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hiprand_kernel<state_type>),
        dim3(4), dim3(64), 0, 0,
        output, output_size
    );
    HIP_CHECK(hipPeekAtLastError());

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

TEST(hiprand_kernel_h_philox4x32_10, hiprand)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_test<state_type>();
}

TEST(hiprand_kernel_h_mrg32k3a, hiprand)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_test<state_type>();
}

TEST(hiprand_kernel_h_xorwow, hiprand)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_test<state_type>();
}

TEST(hiprand_kernel_h_default, hiprand)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_test<state_type>();
}

template<class T>
void hiprand_kernel_h_hiprand_uniform_test()
{
    typedef T state_type;

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hiprand_uniform_kernel<state_type>),
        dim3(4), dim3(64), 0, 0,
        output, output_size
    );
    HIP_CHECK(hipPeekAtLastError());

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
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand_uniform)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_uniform_test<state_type>();
}

TEST(hiprand_kernel_h_mrg32k3a, hiprand_uniform)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_uniform_test<state_type>();
}

TEST(hiprand_kernel_h_xorwow, hiprand_uniform)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_uniform_test<state_type>();
}

TEST(hiprand_kernel_h_default, hiprand_uniform)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_uniform_test<state_type>();
}

template<class T>
void hiprand_kernel_h_hiprand_normal_test()
{
    typedef T state_type;

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hiprand_normal_kernel<state_type>),
        dim3(4), dim3(64), 0, 0,
        output, output_size
    );
    HIP_CHECK(hipPeekAtLastError());

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
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand_normal)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_normal_test<state_type>();
}

TEST(hiprand_kernel_h_mrg32k3a, hiprand_normal)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_normal_test<state_type>();
}

TEST(hiprand_kernel_h_xorwow, hiprand_normal)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_normal_test<state_type>();
}

TEST(hiprand_kernel_h_default, hiprand_normal)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_normal_test<state_type>();
}

template<class T>
void hiprand_kernel_h_hiprand_log_normal_test()
{
    typedef T state_type;

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hiprand_log_normal_kernel<state_type>),
        dim3(4), dim3(64), 0, 0,
        output, output_size
    );
    HIP_CHECK(hipPeekAtLastError());

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
}

TEST(hiprand_kernel_h_philox4x32_10, hiprand_log_normal)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_log_normal_test<state_type>();
}

TEST(hiprand_kernel_h_mrg32k3a, hiprand_log_normal)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_log_normal_test<state_type>();
}

TEST(hiprand_kernel_h_xorwow, hiprand_log_normal)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_log_normal_test<state_type>();
}

TEST(hiprand_kernel_h_default, hiprand_log_normal)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_log_normal_test<state_type>();
}

template<class T>
void hiprand_kernel_h_hiprand_poisson_test(double lambda)
{
    typedef T state_type;

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hiprand_poisson_kernel<state_type>),
        dim3(4), dim3(64), 0, 0,
        output, output_size, lambda
    );
    HIP_CHECK(hipPeekAtLastError());

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
}

template<class T>
void hiprand_kernel_h_hiprand_discrete_test(double lambda)
{
    typedef T state_type;

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hiprandDiscreteDistribution_t discrete_distribution;
    ASSERT_EQ(hiprandCreatePoissonDistribution(lambda, &discrete_distribution), HIPRAND_STATUS_SUCCESS);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hiprand_discrete_kernel<state_type>),
        dim3(4), dim3(64), 0, 0,
        output, output_size, discrete_distribution
    );
    HIP_CHECK(hipPeekAtLastError());

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
    ASSERT_EQ(hiprandDestroyDistribution(discrete_distribution), HIPRAND_STATUS_SUCCESS);

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

const double lambdas[] = { 1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0 };

class hiprand_kernel_h_philox4x32_10_poisson : public ::testing::TestWithParam<double> { };

TEST_P(hiprand_kernel_h_philox4x32_10_poisson, hiprand_poisson)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_poisson_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_philox4x32_10_poisson, hiprand_discrete)
{
    typedef hiprandStatePhilox4_32_10_t state_type;
    hiprand_kernel_h_hiprand_discrete_test<state_type>(GetParam());
}

INSTANTIATE_TEST_CASE_P(hiprand_kernel_h_philox4x32_10_poisson,
                        hiprand_kernel_h_philox4x32_10_poisson,
                        ::testing::ValuesIn(lambdas));

class hiprand_kernel_h_mrg32k3a_poisson : public ::testing::TestWithParam<double> { };

TEST_P(hiprand_kernel_h_mrg32k3a_poisson, hiprand_poisson)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_poisson_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_mrg32k3a_poisson, hiprand_discrete)
{
    typedef hiprandStateMRG32k3a_t state_type;
    hiprand_kernel_h_hiprand_discrete_test<state_type>(GetParam());
}

INSTANTIATE_TEST_CASE_P(hiprand_kernel_h_mrg32k3a_poisson,
                        hiprand_kernel_h_mrg32k3a_poisson,
                        ::testing::ValuesIn(lambdas));

class hiprand_kernel_h_xorwow_poisson : public ::testing::TestWithParam<double> { };

TEST_P(hiprand_kernel_h_xorwow_poisson, hiprand_poisson)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_poisson_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_xorwow_poisson, hiprand_discrete)
{
    typedef hiprandStateXORWOW_t state_type;
    hiprand_kernel_h_hiprand_discrete_test<state_type>(GetParam());
}

INSTANTIATE_TEST_CASE_P(hiprand_kernel_h_xorwow_poisson,
                        hiprand_kernel_h_xorwow_poisson,
                        ::testing::ValuesIn(lambdas));

class hiprand_kernel_h_default_poisson : public ::testing::TestWithParam<double> { };

TEST_P(hiprand_kernel_h_default_poisson, hiprand_poisson)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_poisson_test<state_type>(GetParam());
}

TEST_P(hiprand_kernel_h_default_poisson, hiprand_discrete)
{
    typedef hiprandState_t state_type;
    hiprand_kernel_h_hiprand_discrete_test<state_type>(GetParam());
}

INSTANTIATE_TEST_CASE_P(hiprand_kernel_h_default_poisson,
                        hiprand_kernel_h_default_poisson,
                        ::testing::ValuesIn(lambdas));
