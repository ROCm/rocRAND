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

#ifndef ROCRAND_RNG_PHILOX4X32_10_H_
#define ROCRAND_RNG_PHILOX4X32_10_H_

#include <algorithm>
#include <hip/hip_runtime.h>

#include <rocrand.h>

#include "common.hpp"
#include "generator_type.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"

namespace rocrand_host {
namespace detail {

    inline __device__ unsigned int warp_reduce_min(unsigned int val, int size) {
      for (int offset = size/2; offset > 0; offset /= 2) {
        #if defined(__HIP_PLATFORM_NVCC__) && __CUDACC_VER_MAJOR__ >= 9
        unsigned int temp = __shfl_xor_sync(0xffffffff, (int)val, offset);
        #else
        unsigned int temp = __shfl_xor((int)val, offset);
        #endif
        val = (temp < val) ? temp : val;
      }
      return val;
    }

    struct philox4x32_10_device_engine : public ::rocrand_device::philox4x32_10_engine
    {
        typedef ::rocrand_device::philox4x32_10_engine base_type;
        typedef base_type::philox4x32_10_state state_type;

        __forceinline__ __device__ __host__
        philox4x32_10_device_engine() { }

        __forceinline__ __device__ __host__
        philox4x32_10_device_engine(const unsigned long long seed,
                                    const unsigned long long subsequence,
                                    const unsigned long long offset)
            : base_type(seed, subsequence, offset)
        {

        }

        __forceinline__ __device__ __host__
        ~philox4x32_10_device_engine () {}

        __forceinline__ __device__ __host__
        uint4 next4_leap(unsigned int leap)
        {
            uint4 ret = m_state.result;
            this->discard_state(leap);
            m_state.result = this->ten_rounds(m_state.counter, m_state.key);
            return ret;
        }

        // m_state from base class
    };

    __global__
    void init_engines_kernel(philox4x32_10_device_engine * engines,
                             const unsigned long long seed,
                             const unsigned long long offset)
    {
        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        engines[engine_id] = philox4x32_10_device_engine(seed, engine_id, offset);
    }

    template<unsigned int ThreadsPerEngine, class T, class Distribution>
    __global__
    void generate_kernel(philox4x32_10_device_engine * engines,
                         T * data, const size_t n,
                         Distribution distribution)
    {
        constexpr unsigned int input_width = Distribution::input_width;
        constexpr unsigned int output_width = Distribution::output_width;

        static_assert(4 % input_width == 0 && input_width <= 4, "Incorrect input_width");
        constexpr unsigned int output_per_thread = 4 / input_width;
        constexpr unsigned int full_output_width = output_per_thread * output_width;

        using vec_type = aligned_vec_type<T, output_per_thread * output_width>;

        const unsigned int thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const unsigned int engine_id = thread_id/ThreadsPerEngine;
        const unsigned int stride = hipGridDim_x * hipBlockDim_x;
        size_t index = thread_id;

        // Load device engine
        philox4x32_10_device_engine engine = engines[engine_id];
        if(hipThreadIdx_x%ThreadsPerEngine > 0)
        {
            // Skips hipThreadIdx_x%ThreadsPerEngine states
            engine.discard(4 * (hipThreadIdx_x%ThreadsPerEngine));
        }

        unsigned int input[input_width];
        T output[output_per_thread][output_width];

        const uintptr_t uintptr = reinterpret_cast<uintptr_t>(data);
        const size_t misalignment =
            (
                full_output_width - uintptr / sizeof(T) % full_output_width
            ) % full_output_width;
        const unsigned int head_size = min(n, misalignment);
        const unsigned int tail_size = (n - head_size) % full_output_width;
        const size_t vec_n = (n - head_size) / full_output_width;

        // Save multiple values as one vec_type
        vec_type * vec_data = reinterpret_cast<vec_type *>(data + misalignment);
        while(index < vec_n)
        {
            const uint4 v = engine.next4_leap(ThreadsPerEngine);
            const unsigned int vs[4] = { v.x, v.y, v.z, v.w };
            for(unsigned int s = 0; s < output_per_thread; s++)
            {
                for(unsigned int i = 0; i < input_width; i++)
                {
                    input[i] = vs[s * input_width + i];
                }
                distribution(input, output[s]);
            }
            vec_data[index] = *reinterpret_cast<vec_type *>(output);
            // Next position
            index += stride;
        }

        // Find thread with the smallest state of the engine which id is engine_id
        unsigned int index_min = warp_reduce_min(index, ThreadsPerEngine);
        const bool smallest_state = (index == index_min);

        // Check if we need to save head and tail.
        // Those numbers should be generated by the thread that would
        // save next vec_type.
        // If this condition is met, then we know that (index == index_min)
        // is also true for that thread, so we don't need to check that.
        if(index == vec_n)
        {
            // If data is not aligned by sizeof(vec_type)
            if(head_size > 0)
            {
                const uint4 v = engine.next4_leap(ThreadsPerEngine);
                const unsigned int vs[4] = { v.x, v.y, v.z, v.w };
                for(unsigned int s = 0; s < output_per_thread; s++)
                {
                    for(unsigned int i = 0; i < input_width; i++)
                    {
                        input[i] = vs[s * input_width + i];
                    }
                    distribution(input, output[s]);

                    for(unsigned int o = 0; o < output_width; o++)
                    {
                        if(s * output_width + o < head_size)
                        {
                            data[s * output_width + o] = output[s][o];
                        }
                    }
                }
            }

            if(tail_size > 0)
            {
                const uint4 v = engine.next4_leap(ThreadsPerEngine);
                const unsigned int vs[4] = { v.x, v.y, v.z, v.w };
                for(unsigned int s = 0; s < output_per_thread; s++)
                {
                    for(unsigned int i = 0; i < input_width; i++)
                    {
                        input[i] = vs[s * input_width + i];
                    }
                    distribution(input, output[s]);

                    for(unsigned int o = 0; o < output_width; o++)
                    {
                        if(s * output_width + o < tail_size)
                        {
                            data[n - tail_size + s * output_width + o] = output[s][o];
                        }
                    }
                }
            }
        }

        // Save engine
        if(smallest_state)
            engines[engine_id] = engine;
    }

} // end namespace detail
} // end namespace rocrand_host

class rocrand_philox4x32_10 : public rocrand_generator_type<ROCRAND_RNG_PSEUDO_PHILOX4_32_10>
{
    static constexpr unsigned int s_threads_per_engine = 16;

public:
    using base_type = rocrand_generator_type<ROCRAND_RNG_PSEUDO_PHILOX4_32_10>;
    using engine_type = ::rocrand_host::detail::philox4x32_10_device_engine;

    rocrand_philox4x32_10(unsigned long long seed = 0,
                          unsigned long long offset = 0,
                          hipStream_t stream = 0)
        : base_type(seed, offset, stream),
          m_engines_initialized(false), m_engines(NULL),
          m_engines_size(s_threads * s_blocks / s_threads_per_engine)
    {
        // Allocate device random number engines
        auto error = hipMalloc(&m_engines, sizeof(engine_type) * m_engines_size);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
    }

    ~rocrand_philox4x32_10()
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
        if(m_engines_initialized)
            return ROCRAND_STATUS_SUCCESS;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::init_engines_kernel),
            dim3(s_blocks/s_threads_per_engine), dim3(s_threads), 0, m_stream,
            m_engines, m_seed, m_offset
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_engines_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T> >
    rocrand_status generate(T * data, size_t data_size,
                            Distribution distribution = Distribution())
    {
        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_kernel<s_threads_per_engine>),
            dim3(s_blocks), dim3(s_threads), 0, m_stream,
            m_engines, data, data_size, distribution
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
    const size_t m_engines_size;

    const static uint32_t s_threads = 256;
    const static uint32_t s_blocks = 1024;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<> m_poisson;

    // m_seed from base_type
    // m_offset from base_type
};

#endif // ROCRAND_RNG_PHILOX4X32_10_H_
