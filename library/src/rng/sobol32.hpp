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

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__ __host__
#endif

#include <algorithm>
#include <hip/hip_runtime.h>

#include <rocrand.h>
#include <rocrand_kernel.h>
#include <rocrand_sobol_precomputed.h>

#include "generator_type.hpp"
#include "distributions.hpp"

namespace rocrand_host {
namespace detail {

    typedef ::rocrand_device::sobol32_engine sobol32_device_engine;

    __global__
    void init_sobol_engines_kernel(sobol32_device_engine * engines,
                                   unsigned int * vectors,
                                   unsigned long long offset)
    {
        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int vector_dim = (engine_id % SOBOL_DIM) * 32;
        unsigned long long off = offset + engine_id + 1;
        sobol32_device_engine engine = sobol32_device_engine(vectors + vector_dim, off);
        engine.discard();
        engines[engine_id] = engine;
    }

    template<class Type, class Distribution>
    __global__
    void generate_kernel(sobol32_device_engine * engines,
                         Type * data, const size_t n,
                         Distribution distribution)
    {
        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = engine_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        sobol32_device_engine engine = engines[engine_id];

        while(index < n)
        {
            data[index] = distribution(engine());
            // Next position
            index += stride;
        }

        // Save engine with its state
        engines[engine_id] = engine;
    }
    
    template<class RealType, class Distribution>
    __global__
    void generate_normal_kernel(sobol32_device_engine * engines,
                                RealType * data, const size_t n,
                                Distribution distribution)
    {
        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = engine_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        sobol32_device_engine engine = engines[engine_id];

        while(index < n)
        {
            data[index] = distribution(engine());
            // Next position
            index += stride;
        }

        // Save engine with its state
        engines[engine_id] = engine;
    }
    
} // end namespace detail
} // end namespace rocrand_host

class rocrand_sobol32 : public rocrand_generator_type<ROCRAND_RNG_QUASI_SOBOL32>
{
public:
    using base_type = rocrand_generator_type<ROCRAND_RNG_QUASI_SOBOL32>;
    using engine_type = ::rocrand_host::detail::sobol32_device_engine;
    
    rocrand_sobol32(unsigned long long seed = 0,
                    unsigned long long offset = 0,
                    hipStream_t stream = 0)
        : base_type(seed, offset, stream),
          m_engines_initialized(false), m_engines(NULL), m_engines_size(256 * 128)
    {
        // Allocate device random number engines
        auto error = hipMalloc(&m_engines, sizeof(engine_type) * m_engines_size);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
    }

    ~rocrand_sobol32()
    {
        hipFree(m_engines);
    }

    void reset()
    {
        m_engines_initialized = false;
    }

    /// Changes seed to \p seed and resets generator state.
    void set_seed(unsigned long long seed)
    {
        m_seed = seed;
        m_engines_initialized = false;
    }

    void set_offset(unsigned long long offset)
    {
        m_offset = offset;
        m_engines_initialized = false;
    }

    rocrand_status init()
    {
        if (m_engines_initialized)
            return ROCRAND_STATUS_SUCCESS;

        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128;
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 128;
        #endif
        const uint32_t blocks = max_blocks;
        
        unsigned int * m_vector;
        hipMalloc(&m_vector, sizeof(unsigned int) * SOBOL_N);
        hipMemcpy(m_vector, h_sobol32_direction_vectors, sizeof(unsigned int) * SOBOL_N, hipMemcpyHostToDevice);
        hipDeviceSynchronize();

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::init_sobol_engines_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_engines, m_vector, m_offset
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_engines_initialized = true;
        hipFree(m_vector);

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T> >
    rocrand_status generate(T * data, size_t data_size,
                            const Distribution& distribution = Distribution())
    {
        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 128;
        #endif
        const uint32_t blocks = max_blocks;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_engines, data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T * data, size_t n)
    {
        uniform_distribution<T> udistribution;
        return generate(data, n, udistribution);
    }

    template<class T>
    rocrand_status generate_normal(T * data, size_t data_size, T stddev, T mean)
    {
        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 128;
        #endif
        const uint32_t blocks = max_blocks;

        normal_distribution<T> distribution(mean, stddev);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_normal_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_engines, data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_log_normal(T * data, size_t data_size, T stddev, T mean)
    {
        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 128;
        #endif
        const uint32_t blocks = max_blocks;

        log_normal_distribution<T> distribution(mean, stddev);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_normal_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_engines, data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status generate_poisson(unsigned int * data, size_t data_size, double lambda)
    {
        try
        {
            poisson.set_lambda(lambda);
        }
        catch(rocrand_status status)
        {
            return status;
        }
        if (lambda < 1000)
            return generate(data, data_size, poisson.dis);
        else {
            normal_distribution<double> distribution(lambda, sqrt(lambda));
            return generate(data, data_size, distribution);
        }
    }

private:
    bool m_engines_initialized;
    engine_type * m_engines;
    size_t m_engines_size;
    
    poisson_distribution_manager<> poisson;

    // m_seed from base_type
    // m_offset from base_type
};

#endif // ROCRAND_RNG_SOBOL32_H_
