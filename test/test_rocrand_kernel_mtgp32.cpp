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

#define FQUALIFIERS __forceinline__ __host__ __device__
#include <rocrand_kernel.h>
#include <rocrand_mtgp32_11213.h>

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)
#define ROCRAND_CHECK(state) ASSERT_EQ(state, ROCRAND_STATUS_SUCCESS)

template <class GeneratorState>
__global__
void rocrand_kernel(GeneratorState * states, unsigned int * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x;
    unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int stride = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    rocrand_mtgp32_block_copy(&states[state_id], &state);

    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = rocrand(&state);
        if(index < size)
            output[index] = value;
        // Next position
        index += stride;
    }

    // Save engine with its state
    rocrand_mtgp32_block_copy(&state, &states[state_id]);
}

template <class GeneratorState>
__global__
void rocrand_uniform_kernel(GeneratorState * states, float * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x;
    const unsigned int thread_id = hipThreadIdx_x;
    unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int stride = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    if (thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = rocrand_uniform(&state);
        if(index < size)
            output[index] = value;
        // Next position
        index += stride;
    }

    // Save engine with its state
    if (thread_id == 0)
        states[state_id] = state;
}

template <class GeneratorState>
__global__
void rocrand_normal_kernel(GeneratorState * states, float * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x;
    const unsigned int thread_id = hipThreadIdx_x;
    unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int stride = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    if (thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = rocrand_normal(&state);
        if(index < size)
            output[index] = value;
        // Next position
        index += stride;
    }

    // Save engine with its state
    if (thread_id == 0)
        states[state_id] = state;
}

template <class GeneratorState>
__global__
void rocrand_log_normal_kernel(GeneratorState * states, float * output, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x;
    const unsigned int thread_id = hipThreadIdx_x;
    unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int stride = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    if (thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = rocrand_log_normal(&state, 1.6f, 0.25f);
        if(index < size)
            output[index] = value;
        // Next position
        index += stride;
    }

    // Save engine with its state
    if (thread_id == 0)
        states[state_id] = state;
}

template <class GeneratorState>
__global__
void rocrand_poisson_kernel(GeneratorState * states, unsigned int * output, const size_t size, double lambda)
{
    const unsigned int state_id = hipBlockIdx_x;
    const unsigned int thread_id = hipThreadIdx_x;
    unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    unsigned int stride = hipGridDim_x * hipBlockDim_x;

    __shared__ GeneratorState state;
    if (thread_id == 0)
        state = states[state_id];
    __syncthreads();

    const size_t r = size%hipBlockDim_x;
    const size_t size_rounded_up = r == 0 ? size : size + (hipBlockDim_x - r);
    while(index < size_rounded_up)
    {
        auto value = rocrand_poisson(&state, lambda);
        if(index < size)
            output[index] = value;
        // Next position
        index += stride;
    }

    // Save engine with its state
    if (thread_id == 0)
        states[state_id] = state;
}

TEST(rocrand_kernel_mtgp32, rocrand_state_mtgp32_type)
{
    EXPECT_EQ(sizeof(rocrand_state_mtgp32), 1078 * sizeof(unsigned int));
    EXPECT_EQ(sizeof(rocrand_state_mtgp32[32]), 32 * sizeof(rocrand_state_mtgp32));
}

TEST(rocrand_kernel_mtgp32, rocrand)
{
    typedef rocrand_state_mtgp32 state_type;

    state_type * states;
    hipMalloc(&states, sizeof(state_type) * 8);

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_kernel<state_type>),
        dim3(8), dim3(256), 0, 0,
        states, output, output_size
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
    HIP_CHECK(hipFree(states));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v) / UINT_MAX;
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_mtgp32, rocrand_uniform)
{
    typedef rocrand_state_mtgp32 state_type;

    state_type * states;
    hipMalloc(&states, sizeof(state_type) * 8);

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_uniform_kernel<state_type>),
        dim3(8), dim3(256), 0, 0,
        states, output, output_size
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
    HIP_CHECK(hipFree(states));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_mtgp32, rocrand_normal)
{
    typedef rocrand_state_mtgp32 state_type;

    state_type * states;
    hipMalloc(&states, sizeof(state_type) * 8);

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_normal_kernel<state_type>),
        dim3(8), dim3(256), 0, 0,
        states, output, output_size
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
    HIP_CHECK(hipFree(states));

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

TEST(rocrand_kernel_mtgp32, rocrand_log_normal)
{
    typedef rocrand_state_mtgp32 state_type;

    state_type * states;
    hipMalloc(&states, sizeof(state_type) * 8);

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_log_normal_kernel<state_type>),
        dim3(8), dim3(256), 0, 0,
        states, output, output_size
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
    HIP_CHECK(hipFree(states));

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

class rocrand_kernel_mtgp32_poisson : public ::testing::TestWithParam<double> { };

TEST_P(rocrand_kernel_mtgp32_poisson, rocrand_poisson)
{
    typedef rocrand_state_mtgp32 state_type;

    const double lambda = GetParam();

    state_type * states;
    hipMalloc(&states, sizeof(state_type) * 8);

    ROCRAND_CHECK(rocrand_make_state_mtgp32(states, mtgp32dc_params_fast_11213, 8, 0));

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_poisson_kernel<state_type>),
        dim3(8), dim3(256), 0, 0,
        states, output, output_size, lambda
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
    HIP_CHECK(hipFree(states));

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

INSTANTIATE_TEST_CASE_P(rocrand_kernel_mtgp32_poisson,
                        rocrand_kernel_mtgp32_poisson,
                        ::testing::ValuesIn(lambdas));
