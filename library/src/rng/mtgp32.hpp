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
 * Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University.  All rights reserved.
 * Copyright (c) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
 * University and University of Tokyo.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of the Hiroshima University nor the names of
 *       its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ROCRAND_RNG_MTGP32_H_
#define ROCRAND_RNG_MTGP32_H_

#include <algorithm>
#include <hip/hip_runtime.h>

#include <rocrand.h>
#include <rocrand_mtgp32_11213.h>

#include "generator_type.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"

namespace rocrand_host {
namespace detail {

    typedef ::rocrand_device::mtgp32_engine mtgp32_device_engine;
    typedef ::rocrand_device::mtgp32_state mtgp32_state;

    template<class Type, class Distribution>
    __global__
    void generate_kernel(mtgp32_device_engine * engines,
                         Type * data,
                         const size_t size,
                         const size_t size_up, // size rounded up to the nearest multiple of hipBlockDim_x
                         const size_t size_down, // size rounded down to the nearest multiple of hipBlockDim_x
                         Distribution distribution)
    {
        const unsigned int engine_id = hipBlockIdx_x;
        unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        __shared__ mtgp32_device_engine engine;
        engine.copy(&engines[engine_id]);

        while(index < size_down)
        {
            data[index] = distribution(engine());
            // Next position
            index += stride;
        }
        while(index < size_up)
        {
            auto value = distribution(engine());
            if(index < size)
                data[index] = value;
            // Next position
            index += stride;
        }

        // Save engine with its state
        engines[engine_id].copy(&engine);
    }

} // end namespace detail
} // end namespace rocrand_host

class rocrand_mtgp32 : public rocrand_generator_type<ROCRAND_RNG_PSEUDO_MTGP32>
{
public:
    using base_type = rocrand_generator_type<ROCRAND_RNG_PSEUDO_MTGP32>;
    using engine_type = ::rocrand_host::detail::mtgp32_device_engine;

    rocrand_mtgp32(unsigned long long seed = 0,
                   unsigned long long offset = 0,
                   hipStream_t stream = 0)
        : base_type(seed, offset, stream),
          m_engines_initialized(false), m_engines(NULL), m_engines_size(s_blocks)
    {
        // Allocate device random number engines
        auto error = hipMalloc(&m_engines, sizeof(engine_type) * m_engines_size);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
    }

    ~rocrand_mtgp32()
    {
        hipFree(m_engines);
    }

    void reset()
    {
        m_engines_initialized = false;
    }

    /// Changes seed to \p seed and resets generator state.
    ///
    /// New seed value should not be zero. If \p seed_value is equal
    /// zero, value \p rocrand_mtgp32_DEFAULT_SEED is used instead.
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

        rocrand_status status;

        status = rocrand_make_state_mtgp32(m_engines, mtgp32dc_params_fast_11213, m_engines_size, m_seed);
        if(status != ROCRAND_STATUS_SUCCESS)
            return ROCRAND_STATUS_ALLOCATION_FAILED;

        m_engines_initialized = true;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T> >
    rocrand_status generate(T * data, size_t data_size,
                            const Distribution& distribution = Distribution())
    {
        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        const size_t remainder_value = data_size%s_threads;
        const size_t size_rounded_down = data_size - remainder_value;
        // if remainder is 0, then data_size is a multiple of s_threads, and
        // in this case size_rounded_up must be data_size
        const size_t size_rounded_up =
            remainder_value == 0 ? data_size : size_rounded_down + s_threads;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_kernel),
            dim3(s_blocks), dim3(s_threads), 0, m_stream,
            m_engines, data, data_size, size_rounded_up,
            size_rounded_down, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

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
    bool m_engines_initialized;
    engine_type * m_engines;
    size_t m_engines_size;
    #ifdef __HIP_PLATFORM_NVCC__
    static const uint32_t s_threads = 256;
    static const uint32_t s_blocks = 64;
    #else
    static const uint32_t s_threads = 256;
    static const uint32_t s_blocks = 512;
    #endif

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<> m_poisson;

    // m_seed from base_type
    // m_offset from base_type
};

#endif // ROCRAND_RNG_MTGP32_H_
