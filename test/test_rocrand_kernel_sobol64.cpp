// Copyright (c) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.
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
#include <type_traits>

#include <hip/hip_runtime.h>

#define FQUALIFIERS __forceinline__ __host__ __device__
#include <rocrand/rocrand_kernel.h>
#include <rocrand/rocrand_sobol64_precomputed.h>

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)
#define ROCRAND_CHECK(state) ASSERT_EQ(state, ROCRAND_STATUS_SUCCESS)

template <class GeneratorState>
__global__
__launch_bounds__(32)
void rocrand_init_kernel(GeneratorState * states,
                         const size_t states_size,
                         unsigned long long int * vectors,
                         unsigned long long int offset)
{
    const unsigned int state_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(state_id < states_size)
    {
        GeneratorState state;
        rocrand_init(vectors, offset, &state);
        states[state_id] = state;
    }
}

template <class GeneratorState>
__global__
__launch_bounds__(32)
void rocrand_kernel(unsigned long long int * output, unsigned long long int * vectors, const size_t size)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

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
__launch_bounds__(32)
void rocrand_uniform_kernel(double * output, unsigned long long int * vectors, const size_t size)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    const unsigned int n = size / global_size;
    GeneratorState state;
    rocrand_init(vectors, 1234 + state_id * n, &state);

    for (unsigned int i = 0; i < n; i++)
    {
        output[state_id * n + i] = rocrand_uniform_double(&state);
    }
}

template <class GeneratorState>
__global__
__launch_bounds__(32)
void rocrand_normal_kernel(double * output, unsigned long long int * vectors, const size_t size)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    const unsigned int n = size / global_size;
    GeneratorState state;
    rocrand_init(vectors, 1234 + state_id * n, &state);

    for (unsigned int i = 0; i < n; i++)
    {
        output[state_id * n + i] = rocrand_normal_double(&state);
    }
}

template <class GeneratorState>
__global__
__launch_bounds__(32)
void rocrand_log_normal_kernel(double * output, unsigned long long int * vectors, const size_t size)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    const unsigned int n = size / global_size;
    GeneratorState state;
    rocrand_init(vectors, 1234 + state_id * n, &state);

    for (unsigned int i = 0; i < n; i++)
    {
        output[state_id * n + i] = rocrand_log_normal_double(&state, 1.6f, 0.25f);
    }
}

template <class GeneratorState>
__global__
__launch_bounds__(32)
void rocrand_poisson_kernel(unsigned int * output, unsigned long long int * vectors, const size_t size, double lambda)
{
    const unsigned int state_id    = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int global_size = gridDim.x * blockDim.x;

    const unsigned int n = size / global_size;
    GeneratorState state;
    rocrand_init(vectors, 1234 + state_id * n, &state);

    for (unsigned int i = 0; i < n; i++)
    {
        output[state_id * n + i] = rocrand_poisson(&state, lambda);
    }
}

TEST(rocrand_kernel_sobol64, rocrand_state_sobol64_type)
{
    typedef rocrand_state_sobol64 state_type;
    EXPECT_EQ(sizeof(state_type), 66 * sizeof(unsigned long long int));
    EXPECT_EQ(sizeof(state_type[32]), 32 * sizeof(state_type));
    EXPECT_TRUE(std::is_trivially_copyable<state_type>::value);
    EXPECT_TRUE(std::is_trivially_destructible<state_type>::value);
}

TEST(rocrand_kernel_sobol64, rocrand)
{
    typedef rocrand_state_sobol64 state_type;
    using Type = unsigned long long int;

    const size_t output_size = 8192;
    Type * output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&output), output_size * sizeof(Type)));
    HIP_CHECK(hipDeviceSynchronize());

    Type * m_vector;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&m_vector), sizeof(Type) * 8 * 64));
    HIP_CHECK(hipMemcpy(m_vector, h_sobol64_direction_vectors, sizeof(Type) * 8 * 64, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_kernel<state_type>),
        dim3(8), dim3(32), 0, 0,
        output, m_vector, output_size
    );
    HIP_CHECK(hipGetLastError());

    std::vector<Type> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(Type),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(m_vector));

    double mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<double>(v) / static_cast<double>(UINT64_MAX);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.5, 0.1);
}

TEST(rocrand_kernel_sobol64, rocrand_uniform)
{
    typedef rocrand_state_sobol64 state_type;
    typedef double Type;

    const size_t output_size = 256;
    Type * output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&output), output_size * sizeof(Type)));
    HIP_CHECK(hipDeviceSynchronize());

    typedef unsigned long long int DirectionVectorType;
    DirectionVectorType * m_vector;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&m_vector), sizeof(DirectionVectorType) * 8 * 64));
    HIP_CHECK(hipMemcpy(m_vector, h_sobol64_direction_vectors, sizeof(DirectionVectorType) * 8 * 64, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_uniform_kernel<state_type>),
        dim3(8), dim3(32), 0, 0,
        output, m_vector, output_size
    );
    HIP_CHECK(hipGetLastError());

    std::vector<Type> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(Type),
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

TEST(rocrand_kernel_sobol64, rocrand_normal)
{
    typedef rocrand_state_sobol64 state_type;
    typedef double Type;

    const size_t output_size = 8192;
    Type * output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&output), output_size * sizeof(Type)));
    HIP_CHECK(hipDeviceSynchronize());

    typedef unsigned long long int DirectionVectorType;
    DirectionVectorType * m_vector;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&m_vector), sizeof(DirectionVectorType) * 8 * 64));
    HIP_CHECK(hipMemcpy(m_vector, h_sobol64_direction_vectors, sizeof(DirectionVectorType) * 8 * 64, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_normal_kernel<state_type>),
        dim3(8), dim3(32), 0, 0,
        output, m_vector, output_size
    );
    HIP_CHECK(hipGetLastError());

    std::vector<Type> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(Type),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(m_vector));

    Type mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<Type>(v);
    }
    mean = mean / output_size;
    EXPECT_NEAR(mean, 0.0, 0.2);

    Type stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(static_cast<Type>(v) - mean, 2);
    }
    stddev = stddev / output_size;
    EXPECT_NEAR(stddev, 1.0, 0.2);
}

TEST(rocrand_kernel_sobol64, rocrand_log_normal)
{
    typedef rocrand_state_sobol64 state_type;
    typedef double Type;

    const size_t output_size = 8192;
    Type * output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&output), output_size * sizeof(Type)));
    HIP_CHECK(hipDeviceSynchronize());

    typedef unsigned long long int DirectionVectorType;
    DirectionVectorType * m_vector;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&m_vector), sizeof(DirectionVectorType) * 8 * 64));
    HIP_CHECK(hipMemcpy(m_vector, h_sobol64_direction_vectors, sizeof(DirectionVectorType) * 8 * 64, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_log_normal_kernel<state_type>),
        dim3(8), dim3(32), 0, 0,
        output, m_vector, output_size
    );
    HIP_CHECK(hipGetLastError());

    std::vector<Type> output_host(output_size);
    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(Type),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));
    HIP_CHECK(hipFree(m_vector));

    Type mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<Type>(v);
    }
    mean = mean / output_size;

    Type stddev = 0;
    for(auto v : output_host)
    {
        stddev += std::pow(v - mean, 2);
    }
    stddev = std::sqrt(stddev / output_size);

    Type logmean = std::log(mean * mean / std::sqrt(stddev + mean * mean));
    Type logstd = std::sqrt(std::log(1.0f + stddev/(mean * mean)));

    EXPECT_NEAR(1.6, logmean, 1.6 * 0.2);
    EXPECT_NEAR(0.25, logstd, 0.25 * 0.2);
}

class rocrand_kernel_sobol64_poisson : public ::testing::TestWithParam<double> { };

TEST_P(rocrand_kernel_sobol64_poisson, rocrand_poisson)
{
    typedef rocrand_state_sobol64 state_type;
    typedef double Type;

    const Type lambda = GetParam();

    typedef unsigned long long int DirectionVectorType;
    DirectionVectorType * m_vector;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&m_vector), sizeof(DirectionVectorType) * 8 * 64));
    HIP_CHECK(hipMemcpy(m_vector, h_sobol64_direction_vectors, sizeof(DirectionVectorType) * 8 * 64, hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    const size_t output_size = 8192;
    unsigned int * output;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&output), output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(rocrand_poisson_kernel<state_type>),
        dim3(8), dim3(32), 0, 0,
        output, m_vector, output_size, lambda
    );
    HIP_CHECK(hipGetLastError());

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

    Type mean = 0;
    for(auto v : output_host)
    {
        mean += static_cast<Type>(v);
    }
    mean = mean / output_size;

    Type variance = 0;
    for(auto v : output_host)
    {
        variance += std::pow(v - mean, 2);
    }
    variance = variance / output_size;

    EXPECT_NEAR(mean, lambda, std::max(1.0, lambda * 1e-1));
    EXPECT_NEAR(variance, lambda, std::max(1.0, lambda * 1e-1));
}

const double lambdas[] = { 1.0, 5.5, 20.0, 100.0, 1234.5, 5000.0 };

INSTANTIATE_TEST_SUITE_P(rocrand_kernel_sobol64_poisson,
                        rocrand_kernel_sobol64_poisson,
                        ::testing::ValuesIn(lambdas));
