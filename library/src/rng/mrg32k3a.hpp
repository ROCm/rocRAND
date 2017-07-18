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

/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ROCRAND_RNG_MRG32K3A_H_
#define ROCRAND_RNG_MRG32K3A_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__ __host__
#endif

#include <algorithm>
#include <hip/hip_runtime.h>

#include <rocrand.h>
#include <rocrand_kernel.h>

#include "generator_type.hpp"
#include "distributions.hpp"

namespace rocrand_host {
namespace detail {

    struct mrg32k3a_device_engine : public ::rocrand_device::mrg32k3a_engine
    {
        typedef ::rocrand_device::mrg32k3a_engine base_type;
        typedef base_type::mrg32k3a_state state_type;

        __forceinline__ __device__ __host__
        mrg32k3a_device_engine() { }

        __forceinline__ __device__ __host__
        mrg32k3a_device_engine(const unsigned long long seed,
                               const unsigned long long subsequence,
                               const unsigned long long offset)
            : base_type(seed, subsequence, offset)
        {

        }

        __forceinline__ __device__ __host__
        ~mrg32k3a_device_engine () {}

        // m_state from base class
    };

    template<class Type, class Distribution>
    __global__
    void generate_kernel(mrg32k3a_device_engine * engines,
                         bool init_engines,
                         unsigned long long seed,
                         unsigned long long offset,
                         Type * data, const size_t n,
                         Distribution distribution)
    {
        typedef mrg32k3a_device_engine DeviceEngineType;

        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = engine_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load or init device engine
        DeviceEngineType engine;
        if(init_engines)
        {
            engine = DeviceEngineType(seed, index, 0);
        }
        else
        {
            engine = engines[engine_id];
        }

        // TODO: It's possible to improve performance for situations when
        // generate_poisson_kernel was not called before generate_kernel
        // TODO: We need to check if ordering is so imporant, or if we can
        // skip some random numbers (which increases performance).
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
    void generate_normal_kernel(mrg32k3a_device_engine * engines,
                                bool init_engines,
                                unsigned long long seed,
                                unsigned long long offset,
                                RealType * data, const size_t n,
                                Distribution distribution)
    {
        typedef mrg32k3a_device_engine DeviceEngineType;
        typedef decltype(distribution(engines->next(), engines->next())) RealType2;

        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = engine_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load or init device engine
        DeviceEngineType engine;
        if(init_engines)
        {
            engine = DeviceEngineType(seed, index, 0);
        }
        else
        {
            engine = engines[engine_id];
        }

        RealType2 * data2 = (RealType2 *)data;
        while(index < (n / 2))
        {
            data2[index] = distribution(engine(), engine());
            // Next position
            index += stride;
        }

        // First work-item saves the tail when n is not a multiple of 2
        auto tail_size = n & 1;
        if(engine_id == 0 && tail_size > 0)
        {
            RealType2 result = distribution(engine(), engine());
            // Save the tail
            data[n - tail_size] = (&result.x)[0]; // .x
            if(tail_size > 1) data[n - tail_size + 1] = (&result.x)[1]; // .y
        }
        // Save engine with its state
        engines[engine_id] = engine;
    }

    template <class Distribution>
    __global__
    void generate_poisson_kernel(mrg32k3a_device_engine * engines,
                                 bool init_engines,
                                 unsigned long long seed,
                                 unsigned long long offset,
                                 unsigned int * data, const size_t n,
                                 Distribution distribution)
    {
        typedef mrg32k3a_device_engine DeviceEngineType;

        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = engine_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load or init device engine
        DeviceEngineType engine;
        if(init_engines)
        {
            engine = DeviceEngineType(seed, index, 0);
        }
        else
        {
            engine = engines[engine_id];
        }

        // TODO: Improve performance.
        while(index < n)
        {
            auto result = distribution(engine);
            data[index] = result;
            index += stride;
        }

        // Save engine with its state
        engines[engine_id] = engine;
    }

} // end namespace detail
} // end namespace rocrand_host

class rocrand_mrg32k3a : public rocrand_generator_type<ROCRAND_RNG_PSEUDO_MRG32K3A>
{
public:
    using base_type = rocrand_generator_type<ROCRAND_RNG_PSEUDO_MRG32K3A>;
    using engine_type = ::rocrand_host::detail::mrg32k3a_device_engine;

    rocrand_mrg32k3a(unsigned long long seed = 12345,
                     unsigned long long offset = 1,
                     hipStream_t stream = 0)
        : base_type(seed, offset, stream),
          m_engines_initialized(false), m_engines(NULL), m_engines_size(1024 * 256)
    {
        // Allocate device random number engines
        auto error = hipMalloc(&m_engines, sizeof(engine_type) * m_engines_size);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
    }

    ~rocrand_mrg32k3a()
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

    template<class T, class Distribution = mrg_uniform_distribution<T> >
    rocrand_status generate(T * data, size_t data_size,
                            const Distribution& distribution = Distribution())
    {
        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 1024;
        #endif
        const uint32_t blocks = max_blocks;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_engines, !m_engines_initialized, m_seed, m_offset,
            data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_engines_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T * data, size_t n)
    {
        mrg_uniform_distribution<T> udistribution;
        return generate(data, n, udistribution);
    }

    template<class T>
    rocrand_status generate_normal(T * data, size_t data_size, T stddev, T mean)
    {
        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 1024;
        #endif
        const uint32_t blocks = max_blocks;

        mrg_normal_distribution<T> distribution(mean, stddev);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_normal_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_engines, !m_engines_initialized, m_seed, m_offset,
            data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_engines_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_log_normal(T * data, size_t data_size, T stddev, T mean)
    {
        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 1024;
        #endif
        const uint32_t blocks = max_blocks;

        mrg_log_normal_distribution<T> distribution(mean, stddev);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_normal_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_engines, !m_engines_initialized, m_seed, m_offset,
            data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_engines_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status generate_poisson(unsigned int * data, size_t data_size, double lambda)
    {
        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 1024;
        #endif
        const uint32_t blocks = max_blocks;

        mrg_poisson_distribution<unsigned int> distribution(lambda);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_poisson_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_engines, !m_engines_initialized, m_seed, m_offset,
            data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_engines_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

private:
    bool m_engines_initialized;
    engine_type * m_engines;
    size_t m_engines_size;

    // m_seed from base_type
    // m_offset from base_type
};

#endif // ROCRAND_RNG_MRG32K3A_H_
