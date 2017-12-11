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

#ifndef ROCRAND_RNG_SOBOL32_H_
#define ROCRAND_RNG_SOBOL32_H_

#include <algorithm>
#include <hip/hip_runtime.h>

#include <rocrand.h>
#include <rocrand_sobol_precomputed.h>

#include "generator_type.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"

namespace rocrand_host {
namespace detail {

    typedef ::rocrand_device::sobol32_engine<true> sobol32_device_engine;

    template<class Type, class Distribution>
    __global__
    void generate_kernel(Type * data, const size_t n,
                         const unsigned int * direction_vectors,
                         const unsigned int offset,
                         Distribution distribution)
    {
        const unsigned int dimension = hipBlockIdx_y;
        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Each thread of the current block use the same direction vectors
        // (the dimension is determined by hipBlockIdx_y)
        __shared__ unsigned int vectors[32];
        if (hipThreadIdx_x < 32)
        {
            vectors[hipThreadIdx_x] = direction_vectors[dimension * 32 + hipThreadIdx_x];
        }
        __syncthreads();

        sobol32_device_engine engine(vectors, offset + engine_id);

        const unsigned int start = dimension * n;
        unsigned int index = engine_id;
        while(index < n)
        {
            data[start + index] = distribution(engine.current());
            engine.discard_stride(stride);
            index += stride;
        }
    }

} // end namespace detail
} // end namespace rocrand_host

class rocrand_sobol32 : public rocrand_generator_type<ROCRAND_RNG_QUASI_SOBOL32>
{
public:
    using base_type = rocrand_generator_type<ROCRAND_RNG_QUASI_SOBOL32>;
    using engine_type = ::rocrand_host::detail::sobol32_device_engine;

    rocrand_sobol32(unsigned long long offset = 0,
                    hipStream_t stream = 0)
        : base_type(0, offset, stream),
          m_initialized(false),
          m_dimensions(1)
    {
        // Allocate direction vectors
        hipError_t error;
        error = hipMalloc(&m_direction_vectors, sizeof(unsigned int) * SOBOL_N);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
        error = hipMemcpy(m_direction_vectors, h_sobol32_direction_vectors, sizeof(unsigned int) * SOBOL_N, hipMemcpyHostToDevice);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_INTERNAL_ERROR;
        }
    }

    ~rocrand_sobol32()
    {
        hipFree(m_direction_vectors);
    }

    void reset()
    {
        m_initialized = false;
    }

    void set_offset(unsigned long long offset)
    {
        m_offset = offset;
        m_initialized = false;
    }

    void set_dimensions(unsigned int dimensions)
    {
        m_dimensions = dimensions;
        m_initialized = false;
    }

    rocrand_status init()
    {
        if (m_initialized)
            return ROCRAND_STATUS_SUCCESS;

        m_current_offset = static_cast<unsigned int>(m_offset);
        m_initialized = true;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T> >
    rocrand_status generate(T * data, size_t data_size,
                            const Distribution& distribution = Distribution())
    {
        if (data_size % m_dimensions != 0)
            return ROCRAND_STATUS_LENGTH_NOT_MULTIPLE;

        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 64;
        const uint32_t max_blocks = 4096;
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 4096;
        #endif

        const size_t size = data_size / m_dimensions;
        const uint32_t blocks = std::min(max_blocks, static_cast<uint32_t>((size + threads - 1) / threads));

        // blocks_x must be power of 2 because strided discard (leap frog)
        // supports only power of 2 jumps
        const uint32_t blocks_x = next_power2((blocks + m_dimensions - 1) / m_dimensions);
        const uint32_t blocks_y = m_dimensions;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_kernel),
            dim3(blocks_x, blocks_y), dim3(threads), 0, m_stream,
            data, size,
            static_cast<const unsigned int*>(m_direction_vectors), m_current_offset,
            distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_current_offset += size;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T * data, size_t data_size)
    {
        uniform_distribution<T> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T * data, size_t data_size, T mean, T stddev)
    {
        normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T * data, size_t data_size, T mean, T stddev)
    {
        log_normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    rocrand_status generate_poisson(unsigned int * data, size_t data_size, double lambda)
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
    bool m_initialized;
    unsigned int m_dimensions;
    unsigned int m_current_offset;
    unsigned int * m_direction_vectors;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<ROCRAND_DISCRETE_METHOD_CDF> m_poisson;

    // m_offset from base_type

    size_t next_power2(size_t x)
    {
        size_t power = 1;
        while (power < x)
        {
            power *= 2;
        }
        return power;
    }
};

#endif // ROCRAND_RNG_SOBOL32_H_
