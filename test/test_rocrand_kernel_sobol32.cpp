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
#include <rocrand_sobol_precomputed.h>

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)
#define ROCRAND_CHECK(state) ASSERT_EQ(state, ROCRAND_STATUS_SUCCESS)

template <class GeneratorState>
__global__
void rocrand_init_kernel(GeneratorState * states,
                         const size_t states_size,
                         unsigned int * vectors,
                         unsigned long long offset)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    if(state_id < states_size)
    {
        GeneratorState state;
        rocrand_init(vectors, offset, &state);
        states[state_id] = state;
    }
}

template <class GeneratorState>
__global__
void rocrand_kernel(unsigned int * output, unsigned int * vectors, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    const unsigned int n = size / global_size;
    GeneratorState state;
    rocrand_init(vectors, 1234 + state_id * n, &state);

    for (unsigned int i = 0; i < n; i++)
    {
        output[state_id * n + i] = rocrand(&state);
    }
}

template <class GeneratorState>
__global__
void rocrand_uniform_kernel(float * output, unsigned int * vectors, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    const unsigned int n = size / global_size;
    GeneratorState state;
    rocrand_init(vectors, 1234 + state_id * n, &state);

    for (unsigned int i = 0; i < n; i++)
    {
        output[state_id * n + i] = rocrand_uniform(&state);
    }
}

template <class GeneratorState>
__global__
void rocrand_normal_kernel(float * output, unsigned int * vectors, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    const unsigned int n = size / global_size;
    GeneratorState state;
    rocrand_init(vectors, 1234 + state_id * n, &state);

    for (unsigned int i = 0; i < n; i++)
    {
        output[state_id * n + i] = rocrand_normal(&state);
    }
}

template <class GeneratorState>
__global__
void rocrand_log_normal_kernel(float * output, unsigned int * vectors, const size_t size)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    const unsigned int n = size / global_size;
    GeneratorState state;
    rocrand_init(vectors, 1234 + state_id * n, &state);

    for (unsigned int i = 0; i < n; i++)
    {
        output[state_id * n + i] = rocrand_log_normal(&state, 1.6f, 0.25f);
    }
}

template <class GeneratorState>
__global__
void rocrand_poisson_kernel(unsigned int * output, unsigned int * vectors, const size_t size, double lambda)
{
    const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int global_size = hipGridDim_x * hipBlockDim_x;

    const unsigned int n = size / global_size;
    GeneratorState state;
    rocrand_init(vectors, 1234 + state_id * n, &state);

    for (unsigned int i = 0; i < n; i++)
    {
        output[state_id * n + i] = rocrand_poisson(&state, lambda);
    }
}

TEST(rocrand_kernel_sobol32, rocrand_state_sobol32_type)
{
    EXPECT_EQ(sizeof(rocrand_state_sobol32), 34 * sizeof(unsigned int));
    EXPECT_EQ(sizeof(rocrand_state_sobol32[32]), 32 * sizeof(rocrand_state_sobol32));
}

TEST(rocrand_kernel_sobol32, rocrand)
{
    typedef rocrand_state_sobol32 state_type;

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int * m_vector;
    HIP_CHECK(hipMalloc(&m_vector, sizeof(unsigned int) * 8 * 32));
    HIP_CHECK(hipMemcpy(m_vector, h_sobol32_direction_vectors, sizeof(unsigned int) * 8 * 32, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_kernel<state_type>),
        dim3(8), dim3(32), 0, 0,
        output, m_vector, output_size
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
    HIP_CHECK(hipFree(m_vector));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v) / UINT_MAX;
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_sobol32, rocrand_uniform)
{
    typedef rocrand_state_sobol32 state_type;

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int * m_vector;
    HIP_CHECK(hipMalloc(&m_vector, sizeof(unsigned int) * 8 * 32));
    HIP_CHECK(hipMemcpy(m_vector, h_sobol32_direction_vectors, sizeof(unsigned int) * 8 * 32, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_uniform_kernel<state_type>),
        dim3(8), dim3(32), 0, 0,
        output, m_vector, output_size
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
    HIP_CHECK(hipFree(m_vector));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_sobol32, rocrand_normal)
{
    typedef rocrand_state_sobol32 state_type;

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int * m_vector;
    HIP_CHECK(hipMalloc(&m_vector, sizeof(unsigned int) * 8 * 32));
    HIP_CHECK(hipMemcpy(m_vector, h_sobol32_direction_vectors, sizeof(unsigned int) * 8 * 32, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_normal_kernel<state_type>),
        dim3(8), dim3(32), 0, 0,
        output, m_vector, output_size
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
    HIP_CHECK(hipFree(m_vector));

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

TEST(rocrand_kernel_sobol32, rocrand_log_normal)
{
    typedef rocrand_state_sobol32 state_type;

    const size_t output_size = 8192;
    float * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    unsigned int * m_vector;
    HIP_CHECK(hipMalloc(&m_vector, sizeof(unsigned int) * 8 * 32));
    HIP_CHECK(hipMemcpy(m_vector, h_sobol32_direction_vectors, sizeof(unsigned int) * 8 * 32, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_log_normal_kernel<state_type>),
        dim3(8), dim3(32), 0, 0,
        output, m_vector, output_size
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
    HIP_CHECK(hipFree(m_vector));

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

class rocrand_kernel_sobol32_poisson : public ::testing::TestWithParam<double> { };

TEST_P(rocrand_kernel_sobol32_poisson, rocrand_poisson)
{
    typedef rocrand_state_sobol32 state_type;

    const double lambda = GetParam();

    unsigned int * m_vector;
    HIP_CHECK(hipMalloc(&m_vector, sizeof(unsigned int) * 8 * 32));
    HIP_CHECK(hipMemcpy(m_vector, h_sobol32_direction_vectors, sizeof(unsigned int) * 8 * 32, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_poisson_kernel<state_type>),
        dim3(8), dim3(32), 0, 0,
        output, m_vector, output_size, lambda
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
    HIP_CHECK(hipFree(m_vector));

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

INSTANTIATE_TEST_CASE_P(rocrand_kernel_sobol32_poisson,
                        rocrand_kernel_sobol32_poisson,
                        ::testing::ValuesIn(lambdas));
