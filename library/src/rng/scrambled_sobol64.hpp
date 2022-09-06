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

#ifndef ROCRAND_RNG_SCRAMBLED_SOBOL64_H_
#define ROCRAND_RNG_SCRAMBLED_SOBOL64_H_

#include "common.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"
#include "rocrand/rocrand_scrambled_sobol64.h"

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_scrambled_sobol64_constants.h>
#include <rocrand/rocrand_scrambled_sobol64_precomputed.h>

#include <hip/hip_runtime.h>

#include <algorithm>

namespace rocrand_host
{
namespace detail
{

typedef ::rocrand_device::scrambled_sobol64_engine<true> scrambled_sobol64_device_engine;

template<unsigned int OutputPerThread, class T, class Distribution>
ROCRAND_KERNEL __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE) void generate_kernel_64(
    T*                            data,
    const size_t                  n,
    const unsigned long long int* direction_vectors,
    const unsigned long long int* scramble_constants,
    const unsigned int            offset,
    Distribution                  distribution)
{
    constexpr unsigned int output_per_thread = OutputPerThread;
    using vec_type                           = aligned_vec_type<T, output_per_thread>;

    const unsigned int dimension = hipBlockIdx_y;
    const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int stride    = hipGridDim_x * hipBlockDim_x;
    size_t             index     = engine_id;

    // Each thread of the current block uses the same direction vectors and scrambling constant
    // (the dimension is determined by hipBlockIdx_y)
    __shared__ unsigned long long int vectors[64];
    if(hipThreadIdx_x < 64)
    {
        vectors[hipThreadIdx_x] = direction_vectors[dimension * 64 + hipThreadIdx_x];
    }
    __syncthreads();
    const unsigned int scramble_constant = scramble_constants[dimension];

    data += dimension * n;

    // All distributions generate one output from one input
    // Generation of, for example, 2 shorts from 1 uint or
    // 2 floats from 2 uints using Box-Muller transformation
    // is impossible because the resulting sequence is not
    // quasi-random anymore.

    if(output_per_thread == 1)
    {
        const unsigned int              engine_offset = engine_id * output_per_thread;
        scrambled_sobol64_device_engine engine(vectors, scramble_constant, offset + engine_offset);

        while(index < n)
        {
            data[index] = distribution(engine.current());
            engine.discard_stride(stride);
            index += stride;
        }
    }
    else
    {
        const uintptr_t uintptr = reinterpret_cast<uintptr_t>(data);
        const size_t    misalignment
            = (output_per_thread - uintptr / sizeof(T) % output_per_thread) % output_per_thread;
        const unsigned int head_size = min(n, misalignment);
        const unsigned int tail_size = (n - head_size) % output_per_thread;
        const size_t       vec_n     = (n - head_size) / output_per_thread;

        const unsigned int engine_offset
            = engine_id * output_per_thread
              + (engine_id == 0 ? 0 : head_size); // The first engine writes head_size values
        scrambled_sobol64_device_engine engine(vectors, scramble_constant, offset + engine_offset);

        if(engine_id == 0)
        {
            // If data is not aligned by sizeof(vec_type)
            for(unsigned int o = 0; o < head_size; o++)
            {
                data[o] = distribution(engine.current());
                engine.discard();
            }
        }

        vec_type* vec_data = reinterpret_cast<vec_type*>(data + misalignment);
        while(index < vec_n)
        {
            scrambled_sobol64_device_engine engine_copy = engine;

            T output[output_per_thread];
            for(unsigned int i = 0; i < output_per_thread; i++)
            {
                output[i] = distribution(engine.current());
                engine.discard();
            }

            vec_data[index] = *reinterpret_cast<vec_type*>(output);

            // Restore from a copy and use fast discard_stride with power of 2 stride
            engine = engine_copy;
            engine.discard_stride(stride * output_per_thread);
            index += stride;
        }

        if(index == vec_n)
        {
            // Fill tail values (up to output_per_thread-1)
            for(unsigned int o = 0; o < tail_size; o++)
            {
                data[n - tail_size + o] = distribution(engine.current());
                engine.discard();
            }
        }
    }
}

} // end namespace detail
} // end namespace rocrand_host

class rocrand_scrambled_sobol64 : public rocrand_generator_type<ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64>
{
public:
    using base_type   = rocrand_generator_type<ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64>;
    using engine_type = ::rocrand_host::detail::scrambled_sobol64_device_engine;

    rocrand_scrambled_sobol64(unsigned long long offset = 0, hipStream_t stream = 0)
        : base_type(0, offset, stream), m_initialized(false), m_dimensions(1)
    {
        // Allocate direction vectors
        hipError_t error;
        error
            = hipMalloc(&m_direction_vectors, sizeof(unsigned long long int) * SCRAMBLED_SOBOL64_N);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
        error = hipMemcpy(m_direction_vectors,
                          h_scrambled_sobol64_direction_vectors,
                          sizeof(unsigned long long int) * SCRAMBLED_SOBOL64_N,
                          hipMemcpyHostToDevice);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_INTERNAL_ERROR;
        }
        // Allocate scramble constants
        error = hipMalloc(&m_scramble_constants,
                          sizeof(unsigned long long int) * SCRAMBLED_SOBOL_DIM);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
        error = hipMemcpy(m_scramble_constants,
                          h_scrambled_sobol64_constants,
                          sizeof(unsigned long long int) * SCRAMBLED_SOBOL_DIM,
                          hipMemcpyHostToDevice);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_INTERNAL_ERROR;
        }
    }

    ~rocrand_scrambled_sobol64()
    {
        hipFree(m_direction_vectors);
        hipFree(m_scramble_constants);
    }

    void reset()
    {
        m_initialized = false;
    }

    void set_offset(unsigned long long offset)
    {
        m_offset      = offset;
        m_initialized = false;
    }

    rocrand_status set_dimensions(unsigned int dimensions)
    {
        if(dimensions < 1 || dimensions > SCRAMBLED_SOBOL_DIM)
        {
            return ROCRAND_STATUS_OUT_OF_RANGE;
        }

        m_dimensions  = dimensions;
        m_initialized = false;

        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status init()
    {
        if(m_initialized)
            return ROCRAND_STATUS_SUCCESS;

        m_current_offset = static_cast<unsigned int>(m_offset);
        m_initialized    = true;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = sobol_uniform_distribution<T>>
    rocrand_status generate(T* data, size_t data_size, Distribution distribution = Distribution())
    {
        constexpr unsigned int output_per_thread
            = sizeof(T) >= sizeof(int) ? 1 : sizeof(int) / sizeof(T);

        if(data_size % m_dimensions != 0)
            return ROCRAND_STATUS_LENGTH_NOT_MULTIPLE;

        rocrand_status status = init();
        if(status != ROCRAND_STATUS_SUCCESS)
            return status;

        constexpr uint32_t threads    = 256;
        constexpr uint32_t max_blocks = 4096;

        const size_t   size             = data_size / m_dimensions;
        const uint32_t output_per_block = threads * output_per_thread;
        const uint32_t blocks
            = std::min(max_blocks,
                       static_cast<uint32_t>((size + output_per_block - 1) / output_per_block));

        // blocks_x must be power of 2 because strided discard (leap frog)
        // supports only power of 2 jumps
        const uint32_t blocks_x = next_power2((blocks + m_dimensions - 1) / m_dimensions);
        const uint32_t blocks_y = m_dimensions;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_kernel_64<output_per_thread>),
            dim3(blocks_x, blocks_y),
            dim3(threads),
            0,
            m_stream,
            data,
            size,
            m_direction_vectors,
            m_scramble_constants,
            m_current_offset,
            distribution);
        // Check kernel status
        if(hipGetLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_current_offset += size;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T* data, size_t data_size)
    {
        sobol_uniform_distribution<T> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T* data, size_t data_size, T mean, T stddev)
    {
        sobol_normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T* data, size_t data_size, T mean, T stddev)
    {
        sobol_log_normal_distribution<T, unsigned long long int> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_poisson(T* data, size_t data_size, double lambda)
    {
        try
        {
            m_poisson.set_lambda(lambda);
        }
        catch(rocrand_status status)
        {
            return status;
        }
        return generate(data, data_size, m_poisson.dis);
    }

private:
    bool                    m_initialized;
    unsigned int            m_dimensions;
    unsigned int            m_current_offset;
    unsigned long long int* m_direction_vectors;
    unsigned long long int* m_scramble_constants;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<ROCRAND_DISCRETE_METHOD_CDF> m_poisson;

    // m_offset from base_type

    size_t next_power2(size_t x)
    {
        size_t power = 1;
        while(power < x)
        {
            power *= 2;
        }
        return power;
    }
};

#endif // ROCRAND_RNG_SCRAMBLED_SOBOL64_H_
