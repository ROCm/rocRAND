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

#include "generator_type.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"

namespace rocrand_host {
namespace detail {

    struct double2_unaligned
    {
        double x;
        double y;
    };

    struct uint4_unaligned
    {
        unsigned int x;
        unsigned int y;
        unsigned int z;
        unsigned int w;
    };

    struct float4_unaligned
    {
        float x;
        float y;
        float z;
        float w;
    };

    struct double4_unaligned
    {
        double x;
        double y;
        double z;
        double w;
    };

    template<class T>
    struct unaligned_type
    {
        typedef void type;
    };

    template<>
    struct unaligned_type<double2>
    {
        typedef double2_unaligned type;
    };

    template<>
    struct unaligned_type<uint4>
    {
        typedef uint4_unaligned type;
    };

    template<>
    struct unaligned_type<float4>
    {
        typedef float4_unaligned type;
    };

    template<>
    struct unaligned_type<double4>
    {
        typedef double4_unaligned type;
    };

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

    template<unsigned int ThreadsPerEngine, class Distribution>
    __global__
    void generate_kernel(philox4x32_10_device_engine * engines,
                         double * data, const size_t n,
                         Distribution distribution)
    {
        typedef philox4x32_10_device_engine DeviceEngineType;
        typedef decltype(distribution(uint4())) Type2;
        typedef typename unaligned_type<Type2>::type Type2_unaligned;

        unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const unsigned int engine_id = index/ThreadsPerEngine;
        const unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        DeviceEngineType engine = engines[engine_id];
        if(hipThreadIdx_x%ThreadsPerEngine > 0)
        {
            // Skips hipThreadIdx_x%ThreadsPerEngine states
            engine.discard(4 * (hipThreadIdx_x%ThreadsPerEngine));
        }

        if(((uintptr_t)data)%(sizeof(Type2)) == 0)
        {
            Type2 * data2 = (Type2 *)data;
            while(index < (n/2))
            {
                data2[index] = distribution(engine.next4_leap(ThreadsPerEngine));
                // Next position
                index += stride;
            }
        }
        else
        {
            Type2_unaligned * data2 = (Type2_unaligned *)data;
            while(index < (n/2))
            {
                Type2 result = distribution(engine.next4_leap(ThreadsPerEngine));
                data2[index] = *(Type2_unaligned*)(&result);  // reinterpret as Type4_unaligned
                // Next position
                index += stride;
            }
        }

        // Find thread with the smallest state of the engine which id is engine_id
        unsigned int index_min = warp_reduce_min(index, ThreadsPerEngine);
        const bool smallest_state = (index == index_min);

        // Check if we need to save tail (last 1 random number).
        // Those numbers should be generated by the thread that would
        // save next uint4 if n was equal n+1 (index < (n/2) would be
        // true in such situation).
        // If this condition is met, then we know that (index == index_min)
        // is also true for that thread, so we don't need to check that.
        auto tail_size = n & 1;
        if((index == n/2) && tail_size > 0)
        {
            Type2 result = distribution(engine.next4());
            // Save the tail
            data[n - tail_size] = result.x;
        }

        // Save engine
        if(smallest_state)
            engines[engine_id] = engine;
    }

    template<unsigned int ThreadsPerEngine, class Type, class Distribution>
    __global__
    void generate_kernel(philox4x32_10_device_engine * engines,
                         Type * data, const size_t n,
                         Distribution distribution)
    {
        typedef philox4x32_10_device_engine DeviceEngineType;
        typedef decltype(distribution(uint4())) Type4;
        typedef typename unaligned_type<Type4>::type Type4_unaligned;

        unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const unsigned int engine_id = index/ThreadsPerEngine;
        const unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        DeviceEngineType engine = engines[engine_id];
        if(hipThreadIdx_x%ThreadsPerEngine > 0)
        {
            // Skips hipThreadIdx_x%ThreadsPerEngine states
            engine.discard(4 * (hipThreadIdx_x%ThreadsPerEngine));
        }

        if(((uintptr_t)data)%(sizeof(Type4)) == 0)
        {
            Type4 * data4 = (Type4 *)data;
            while(index < (n/4))
            {
                data4[index] = distribution(engine.next4_leap(ThreadsPerEngine));
                // Next position
                index += stride;
            }
        }
        else
        {
            Type4_unaligned * data4 = (Type4_unaligned *)data;
            while(index < (n/4))
            {
                Type4 result = distribution(engine.next4_leap(ThreadsPerEngine));
                data4[index] = *(Type4_unaligned*)(&result);  // reinterpret as Type4_unaligned
                // Next position
                index += stride;
            }
        }

        // Find thread with the smallest state of the engine which id is engine_id
        unsigned int index_min = warp_reduce_min(index, ThreadsPerEngine);
        const bool smallest_state = (index == index_min);

        // Check if we need to save tail (last 1,2,3 random number).
        // Those numbers should be generated by the thread that would
        // save next uint4 if n was equal n+3 (index < (n/4) would be
        // true in such situation).
        // If this condition is met, then we know that (index == index_min)
        // is also true for that thread, so we don't need to check that.
        auto tail_size = n & 3;
        if((index == n/4) && tail_size > 0)
        {
            Type4 result = distribution(engine.next4());
            // Save the tail
            data[n - tail_size] = result.x;
            if(tail_size > 1) data[n - tail_size + 1] = result.y;
            if(tail_size > 2) data[n - tail_size + 2] = result.z;
        }

        // Save engine
        if(smallest_state)
            engines[engine_id] = engine;
    }

    template<unsigned int ThreadsPerEngine, class RealType, class Distribution>
    __global__
    void generate_normal_kernel(philox4x32_10_device_engine * engines,
                                RealType * data, const size_t n,
                                Distribution distribution)
    {
        typedef philox4x32_10_device_engine DeviceEngineType;
        // RealTypeX can be float4, double2
        typedef decltype(distribution(uint4())) RealTypeX;
        // x can be 2 or 4
        const unsigned x = sizeof(RealTypeX) / sizeof(RealType);

        unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const unsigned int engine_id = index/ThreadsPerEngine;
        const unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        DeviceEngineType engine = engines[engine_id];
        if(hipThreadIdx_x%ThreadsPerEngine > 0)
        {
            // Skips hipThreadIdx_x%ThreadsPerEngine states
            engine.discard(4 * (hipThreadIdx_x%ThreadsPerEngine));
        }

        RealTypeX * dataX = (RealTypeX *)data;
        while(index < (n/x))
        {
            dataX[index] = distribution(engine.next4_leap(ThreadsPerEngine));
            // Next position
            index += stride;
        }

        // Find thread with the smallest state of the engine which id is engine_id
        unsigned int index_min = warp_reduce_min(index, ThreadsPerEngine);
        const bool smallest_state = index == index_min;

        // Check if we need to save tail (last 1,..,(x-1) random number).
        // Those numbers should be generated by the thread that would
        // save next uint4 if n was equal n+(x-1) (index < (n/x) would be
        // true in such situation).
        // If this condition is met, then we know that (index == index_min)
        // is also true for that thread, so we don't need to check that.
        auto tail_size = n & (x - 1);
        if((index == n/4) && tail_size > 0)
        {
            RealTypeX result = distribution(engine.next4());
            // Save the tail
            data[n - tail_size] = (&result.x)[0]; // .x
            if(tail_size > 1) data[n - tail_size + 1] = (&result.x)[1]; // .y
            if(tail_size > 2) data[n - tail_size + 2] = (&result.x)[2]; // .z
        }

        // Save engine
        if(smallest_state)
            engines[engine_id] = engine;
    }

    template <unsigned int ThreadsPerEngine, class Distribution>
    __global__
    void generate_poisson_kernel(philox4x32_10_device_engine * engines,
                                 unsigned int * data, const size_t n,
                                 const Distribution distribution)
    {
        typedef philox4x32_10_device_engine DeviceEngineType;

        unsigned int index = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        const unsigned int engine_id = index/ThreadsPerEngine;
        const unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        DeviceEngineType engine = engines[engine_id];
        if(hipThreadIdx_x%ThreadsPerEngine > 0)
        {
            // Skips hipThreadIdx_x%ThreadsPerEngine states
            engine.discard(4 * (hipThreadIdx_x%ThreadsPerEngine));
        }

        if(((uintptr_t)data)%(sizeof(uint4)) == 0)
        {
            uint4 * data4 = (uint4 *)data;
            while(index < (n / 4))
            {
                const uint4 u4 = engine.next4();
                const uint4 result = uint4 {
                    distribution(u4.x),
                    distribution(u4.y),
                    distribution(u4.z),
                    distribution(u4.w)
                };
                data4[index] = result;
                index += stride;
            }
        }
        else
        {
            uint4_unaligned * data4 = (uint4_unaligned *)data;
            while(index < (n / 4))
            {
                const uint4 u4 = engine.next4();
                const uint4 result = uint4 {
                    distribution(u4.x),
                    distribution(u4.y),
                    distribution(u4.z),
                    distribution(u4.w)
                };
                data4[index] = *(uint4_unaligned*)(&result); // reinterpret as uint4_unaligned
                index += stride;
            }
        }

        // Find thread with the smallest state of the engine which id is engine_id
        unsigned int index_min = warp_reduce_min(index, ThreadsPerEngine);
        const bool smallest_state = index == index_min;

        // First work-item saves the tail when n is not a multiple of 4
        auto tail_size = n & 3;
        if((index == n/4) == 0 && tail_size > 0)
        {
            const uint4 u4 = engine.next4();
            const uint4 result = uint4 {
                distribution(u4.x),
                distribution(u4.y),
                distribution(u4.z),
                distribution(u4.w)
            };
            data[n - tail_size] = (&result.x)[0]; // .x
            if(tail_size > 1) data[n - tail_size + 1] = (&result.x)[1]; // .y
            if(tail_size > 2) data[n - tail_size + 2] = (&result.x)[2]; // .z
        }

        // Save engine with its state
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
                            const Distribution& distribution = Distribution())
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
        uniform_distribution<T> udistribution;
        return generate(data, data_size, udistribution);
    }

    template<class T>
    rocrand_status generate_normal(T * data, size_t data_size, T mean, T stddev)
    {
        // data_size must be even
        // data must be aligned to 2 * sizeof(T) bytes
        if(data_size%2 != 0 || ((uintptr_t)(data)%(2*sizeof(T))) != 0)
        {
            return ROCRAND_STATUS_LENGTH_NOT_MULTIPLE;
        }

        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        normal_distribution<T> distribution(mean, stddev);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_normal_kernel<s_threads_per_engine>),
            dim3(s_blocks), dim3(s_threads), 0, m_stream,
            m_engines, data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_log_normal(T * data, size_t data_size, T mean, T stddev)
    {
        // data_size must be even
        // data must be aligned to 2 * sizeof(T) bytes
        if(data_size%2 != 0 || ((uintptr_t)(data)%(2*sizeof(T))) != 0)
        {
            return ROCRAND_STATUS_LENGTH_NOT_MULTIPLE;
        }

        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        log_normal_distribution<T> distribution(mean, stddev);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_normal_kernel<s_threads_per_engine>),
            dim3(s_blocks), dim3(s_threads), 0, m_stream,
            m_engines, data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status generate_poisson(unsigned int * data, size_t data_size, double lambda)
    {
        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        try
        {
            m_poisson.set_lambda(lambda);
        }
        catch(rocrand_status status)
        {
            return status;
        }

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_poisson_kernel<s_threads_per_engine>),
            dim3(s_blocks), dim3(s_threads), 0, m_stream,
            m_engines, data, data_size, m_poisson.dis
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        return ROCRAND_STATUS_SUCCESS;
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
