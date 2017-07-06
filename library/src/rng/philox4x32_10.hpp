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

#ifndef FQUALIFIERS
#define FQUALIFIERS __host__ __device__
#endif // FQUALIFIERS

#include <rocrand.h>

#include "philox4x32_10_state.hpp"
#include "generator_type.hpp"
#include "distributions.hpp"

namespace rocrand_philox4x32_10_detail
{
    // HCC
    #ifdef __HIP_DEVICE_COMPILE__
    __device__
    unsigned int mulhilo32(unsigned int x, unsigned int y, unsigned int& z)
    {
        z = __umulhi(x, y);
        return x * y;
    }
    // NVCC
    #elif __CUDA_ARCH__ > 0
    __device__
    unsigned int mulhilo32(unsigned int x, unsigned int y, unsigned int& z)
    {
        z = __umulhi(x, y);
        return x * y;
    }
    #else
    __host__ __device__
    unsigned int mulhilo32(unsigned int x, unsigned int y, unsigned int& z)
    {
        unsigned long long xy =
            static_cast<unsigned long long>(x) * static_cast<unsigned long long>(y);
        z = xy >> 32;
        return static_cast<unsigned int>(xy);
    }
    #endif

    template <
        class Type, class StateType,
        class Generator, class Distribution
    >
    __global__
    void generate_kernel(StateType init_state,
                         Type * data, const size_t n,
                         Generator generator,
                         Distribution distribution)
    {
        unsigned int id = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x) * 4;
        unsigned int stride = hipGridDim_x * hipBlockDim_x * 4;

        StateType state = init_state;
        generator.discard(&state, id);
        while((id + 3) < n)
        {
            auto result = distribution(generator(&state));
            data[id + 0] = result.x;
            data[id + 1] = result.y;
            data[id + 2] = result.z;
            data[id + 3] = result.w;
            generator.discard(&state, stride);
            id += stride;
        }
        if(id < n)
        {
            auto result = distribution(generator(&state));
            data[id + 0] = result.x;
            if((id + 1) < n) data[id + 1] = result.y;
            if((id + 2) < n) data[id + 2] = result.z;
            if((id + 3) < n) data[id + 3] = result.w;
        }
    }

    template <
        class RealType, class StateType,
        class Generator, class Distribution
    >
    __global__
    void generate_normal_kernel(StateType init_state,
                                RealType * data, const size_t n,
                                Generator generator,
                                Distribution distribution)
    {
        // TODO: implement
    }

    template <
        class RealType, class StateType,
        class Generator, class Distribution
    >
    __global__
    void generate_log_normal_kernel(StateType init_state,
                                    RealType * data, const size_t n,
                                    Generator generator,
                                    Distribution distribution)
    {
        // TODO: implement
    }

} // end namespace rocrand_philox4x32_10_detail

class rocrand_philox4x32_10 : public rocrand_generator_type<ROCRAND_RNG_PSEUDO_PHILOX4_32_10>
{
public:
    using base_type = rocrand_generator_type<ROCRAND_RNG_PSEUDO_PHILOX4_32_10>;
    using state_type = base_type::state_type;

    class philox4x32_10_generator
    {
        // Constants from Random123
        // See https://www.deshawresearch.com/resources_random123.html
        static const unsigned int PHILOX_M4x32_0 = 0xD2511F53;
        static const unsigned int PHILOX_M4x32_1 = 0xCD9E8D57;
        static const unsigned int PHILOX_W32_0   = 0x9E3779B9;
        static const unsigned int PHILOX_W32_1   = 0xBB67AE85;

    public:

        inline __host__ __device__
        uint4 operator()(state_type * state)
        {
            return ten_rounds(state->counter, state->key);
        }

        inline __host__ __device__
        void discard(state_type * state, unsigned int n)
        {
            state->discard(n);
        }

        inline __host__ __device__
        void discard(state_type * state)
        {
            state->discard();
        }

    private:
        inline  __host__ __device__
        uint4 single_round(uint4 counter, uint2 key)
        {
            namespace detail = rocrand_philox4x32_10_detail;

            // Source: Random123
            unsigned int hi0;
            unsigned int hi1;
            unsigned int lo0 = detail::mulhilo32(PHILOX_M4x32_0, counter.x, hi0);
            unsigned int lo1 = detail::mulhilo32(PHILOX_M4x32_1, counter.z, hi1);
            return uint4 {
                hi1 ^ counter.y ^ key.x,
                lo1,
                hi0 ^ counter.w ^ key.y,
                lo0
            };
        }

        inline __host__ __device__
        uint2 bumpkey(uint2 key)
        {
            key.x += PHILOX_W32_0;
            key.y += PHILOX_W32_1;
            return key;
        }

        inline __host__ __device__
        uint4 ten_rounds(uint4 counter, uint2 key)
        {
            counter = single_round(counter, key); key = bumpkey(key); // 1
            counter = single_round(counter, key); key = bumpkey(key); // 2
            counter = single_round(counter, key); key = bumpkey(key); // 3
            counter = single_round(counter, key); key = bumpkey(key); // 4
            counter = single_round(counter, key); key = bumpkey(key); // 5
            counter = single_round(counter, key); key = bumpkey(key); // 6
            counter = single_round(counter, key); key = bumpkey(key); // 7
            counter = single_round(counter, key); key = bumpkey(key); // 8
            counter = single_round(counter, key); key = bumpkey(key); // 9
            return single_round(counter, key);                        // 10
        }
    };

    rocrand_philox4x32_10(unsigned long long offset = 0, hipStream_t stream = 0)
        : base_type(offset, stream), m_state(0)
    {

    }

    ~rocrand_philox4x32_10()
    {

    }

    void reset()
    {
        m_state.reset();
    }

    /// Changes seed to \p seed and reset generator state.
    void set_seed(unsigned long long seed)
    {
        m_state.set_seed(seed);
    }

    state_type get_state() const
    {
        return m_state;
    }

    void discard(unsigned int n)
    {
        m_state.discard(n);
    }

    template<class T, class Distribution = uniform_distribution<T> >
    rocrand_status generate(T * data, size_t n,
                            const Distribution& distribution = Distribution())
    {
        const uint32_t threads = 256;
        const uint32_t block_size =
            std::min<uint32_t>(16384, (n + threads - 1) / threads);

        namespace detail = rocrand_philox4x32_10_detail;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::generate_kernel),
            dim3(block_size), dim3(threads), 0, stream,
            m_state,
            data, n,
            m_generator, distribution
        );
        // Progress state
        m_generator.discard(&m_state, ((n + 3) / 4));
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T * data, size_t n)
    {
        uniform_distribution<T> udistribution;
        return generate(data, n, udistribution);
    }

    template<class T>
    rocrand_status generate_normal(T * data, size_t n, T stddev, T mean)
    {
        normal_distribution<T> ndistribution(mean, stddev);
        namespace detail = rocrand_philox4x32_10_detail;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::generate_normal_kernel),
            dim3(1), dim3(n), 0, stream,
            m_state,
            data, n,
            m_generator, ndistribution
        );
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_log_normal(T * data, size_t n, T stddev, T mean)
    {
        log_normal_distribution<T> lndistribution(mean, stddev);
        namespace detail = rocrand_philox4x32_10_detail;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::generate_log_normal_kernel),
            dim3(1), dim3(n), 0, stream,
            m_state,
            data, n,
            m_generator, lndistribution
        );
        return ROCRAND_STATUS_SUCCESS;
    }

private:
    state_type m_state;
    philox4x32_10_generator m_generator;
};

#endif // ROCRAND_RNG_PHILOX4X32_10_H_
