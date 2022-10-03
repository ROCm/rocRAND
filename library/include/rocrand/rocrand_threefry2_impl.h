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

#ifndef ROCRAND_THREEFRY2_IMPL_H_
#define ROCRAND_THREEFRY2_IMPL_H_

#ifndef FQUALIFIERS
    #define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include "rocrand/rocrand_threefry_common.h"

#ifndef THREEFRY2x32_DEFAULT_ROUNDS
    #define THREEFRY2x32_DEFAULT_ROUNDS 20
#endif

#ifndef THREEFRY2x64_DEFAULT_ROUNDS
    #define THREEFRY2x64_DEFAULT_ROUNDS 20
#endif

/* Output from skein_rot_search (srs32x2-X5000.out)
// Random seed = 1. BlockSize = 64 bits. sampleCnt =  1024. rounds =  8, minHW_or=28
// Start: Tue Jul 12 11:11:33 2011
// rMin = 0.334. #0206[*07] [CRC=1D9765C0. hw_OR=32. cnt=16384. blkSize=  64].format   */
static constexpr __device__ int THREEFRY_ROTATION_32_2[] = {13, 15, 26, 6, 17, 29, 16, 24};

/*
// Output from skein_rot_search: (srs64_B64-X1000)
// Random seed = 1. BlockSize = 128 bits. sampleCnt =  1024. rounds =  8, minHW_or=57
// Start: Tue Mar  1 10:07:48 2011
// rMin = 0.136. #0325[*15] [CRC=455A682F. hw_OR=64. cnt=16384. blkSize= 128].format
*/
static constexpr __device__ int THREEFRY_ROTATION_64_2[] = {16, 42, 12, 31, 16, 32, 24, 21};

namespace rocrand_device
{

template<class value>
FQUALIFIERS int threefry_rotation_array(int index);

template<>
FQUALIFIERS int threefry_rotation_array<unsigned int>(int index)
{
    return THREEFRY_ROTATION_32_2[index];
};

template<>
FQUALIFIERS int threefry_rotation_array<unsigned long long>(int index)
{
    return THREEFRY_ROTATION_64_2[index];
};

template<typename state_value, typename value, unsigned int Nrounds>
class threefry_engine2_base
{
public:
    struct threefry_state_2
    {
        state_value  counter;
        state_value  key;
        state_value  result;
        unsigned int substate;
    };

    FQUALIFIERS
    void discard(unsigned long long offset)
    {
        this->discard_impl(offset);
        m_state.result = this->threefry_rounds(m_state.counter, m_state.key);
    }

    FQUALIFIERS
    void discard()
    {
        m_state.result = this->threefry_rounds(m_state.counter, m_state.key);
    }

    /// Advances the internal state to skip \p subsequence subsequences.
    /// A subsequence is 2^33 numbers long for 32 bit and 2^65 for 64 bits generator.
    FQUALIFIERS
    void discard_subsequence(unsigned long long subsequence)
    {
        this->discard_subsequence_impl(subsequence);
        m_state.result = this->threefry_rounds(m_state.counter, m_state.key);
    }

    FQUALIFIERS
    value operator()()
    {
        return this->next();
    }

    FQUALIFIERS
    value next()
    {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
        value ret = m_state.result.data[m_state.substate];
#else
        value ret = (&m_state.result.x)[m_state.substate];
#endif
        m_state.substate++;
        if(m_state.substate == 2)
        {
            m_state.substate = 0;
            m_state.counter  = this->bump_counter(m_state.counter);
            m_state.result   = this->threefry_rounds(m_state.counter, m_state.key);
        }
        return ret;
    }

    FQUALIFIERS
    state_value next2()
    {
        state_value ret = m_state.result;
        m_state.counter = this->bump_counter(m_state.counter);
        m_state.result  = this->threefry_rounds(m_state.counter, m_state.key);

        return this->interleave(ret, m_state.result);
    }

protected:
    FQUALIFIERS
    state_value threefry_rounds(state_value counter, state_value key)
    {
        state_value X;
        value       ks[2 + 1];

        static_assert(Nrounds <= 32, "32 or less only supported in threefry rounds");

        ks[2] = skein_ks_party<value>();

        ks[0] = key.x;
        ks[1] = key.y;

        X.x = counter.x;
        X.y = counter.y;

        ks[2] ^= key.x;
        ks[2] ^= key.y;

        /* Insert initial key before round 0 */
        X.x += ks[0];
        X.y += ks[1];

        if(Nrounds > 0)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0));
            X.y ^= X.x;
        }
        if(Nrounds > 1)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1));
            X.y ^= X.x;
        }
        if(Nrounds > 2)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2));
            X.y ^= X.x;
        }
        if(Nrounds > 3)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3));
            X.y ^= X.x;
        }
        if(Nrounds > 3)
        {
            /* InjectKey(r=1) */
            X.x += ks[1];
            X.y += ks[2];
            X.y += 1; /* X.v[2-1] += r  */
        }
        if(Nrounds > 4)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4));
            X.y ^= X.x;
        }
        if(Nrounds > 5)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5));
            X.y ^= X.x;
        }
        if(Nrounds > 6)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6));
            X.y ^= X.x;
        }
        if(Nrounds > 7)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7));
            X.y ^= X.x;
        }
        if(Nrounds > 7)
        {
            /* InjectKey(r=2) */
            X.x += ks[2];
            X.y += ks[0];
            X.y += 2;
        }
        if(Nrounds > 8)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0));
            X.y ^= X.x;
        }
        if(Nrounds > 9)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1));
            X.y ^= X.x;
        }
        if(Nrounds > 10)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2));
            X.y ^= X.x;
        }
        if(Nrounds > 11)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3));
            X.y ^= X.x;
        }
        if(Nrounds > 11)
        {
            /* InjectKey(r=3) */
            X.x += ks[0];
            X.y += ks[1];
            X.y += 3;
        }
        if(Nrounds > 12)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4));
            X.y ^= X.x;
        }
        if(Nrounds > 13)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5));
            X.y ^= X.x;
        }
        if(Nrounds > 14)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6));
            X.y ^= X.x;
        }
        if(Nrounds > 15)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7));
            X.y ^= X.x;
        }
        if(Nrounds > 15)
        {
            /* InjectKey(r=4) */
            X.x += ks[1];
            X.y += ks[2];
            X.y += 4;
        }
        if(Nrounds > 16)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0));
            X.y ^= X.x;
        }
        if(Nrounds > 17)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1));
            X.y ^= X.x;
        }
        if(Nrounds > 18)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2));
            X.y ^= X.x;
        }
        if(Nrounds > 19)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3));
            X.y ^= X.x;
        }
        if(Nrounds > 19)
        {
            /* InjectKey(r=5) */
            X.x += ks[2];
            X.y += ks[0];
            X.y += 5;
        }
        if(Nrounds > 20)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4));
            X.y ^= X.x;
        }
        if(Nrounds > 21)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5));
            X.y ^= X.x;
        }
        if(Nrounds > 22)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6));
            X.y ^= X.x;
        }
        if(Nrounds > 23)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7));
            X.y ^= X.x;
        }
        if(Nrounds > 23)
        {
            /* InjectKey(r=6) */
            X.x += ks[0];
            X.y += ks[1];
            X.y += 6;
        }
        if(Nrounds > 24)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0));
            X.y ^= X.x;
        }
        if(Nrounds > 25)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1));
            X.y ^= X.x;
        }
        if(Nrounds > 26)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2));
            X.y ^= X.x;
        }
        if(Nrounds > 27)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3));
            X.y ^= X.x;
        }
        if(Nrounds > 27)
        {
            /* InjectKey(r=7) */
            X.x += ks[1];
            X.y += ks[2];
            X.y += 7;
        }
        if(Nrounds > 28)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4));
            X.y ^= X.x;
        }
        if(Nrounds > 29)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5));
            X.y ^= X.x;
        }
        if(Nrounds > 30)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6));
            X.y ^= X.x;
        }
        if(Nrounds > 31)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7));
            X.y ^= X.x;
        }
        if(Nrounds > 31)
        {
            /* InjectKey(r=8) */
            X.x += ks[2];
            X.y += ks[0];
            X.y += 8;
        }
        return X;
    }

    // Advances the internal state to skip \p offset numbers.
    // DOES NOT CALCULATE NEW 4 UINTs (m_state.result)
    FQUALIFIERS
    void discard_impl(unsigned long long offset)
    {
        // Adjust offset for subset
        m_state.substate += offset & 1;
        offset += m_state.substate < 2 ? 0 : 2;
        m_state.substate += m_state.substate < 2 ? 0 : -2;
        // Discard states
        this->discard_state(offset / 2);
    }

    // DOES NOT CALCULATE NEW 4 UINTs (m_state.result)
    FQUALIFIERS
    void discard_subsequence_impl(unsigned long long subsequence)
    {
        m_state.counter.y += subsequence;
    }

    FQUALIFIERS
    void discard_state(unsigned long long offset)
    {
        value old_counter = m_state.counter.x;
        m_state.counter.x += offset;
        m_state.counter.y += offset && m_state.counter.x <= old_counter ? 1 : 0;
    }

    FQUALIFIERS
    static state_value bump_counter(state_value counter)
    {
        counter.x++;
        value add = counter.x == 0 ? 1 : 0;
        counter.y += add;
        return counter;
    }

    FQUALIFIERS
    state_value interleave(const state_value prev, const state_value next) const
    {
        switch(m_state.substate)
        {
            case 0: return prev;
            case 1: return state_value{prev.y, next.x};
        }
        __builtin_unreachable();
    }

protected:
    threefry_state_2 m_state;
}; // threefry_engine2_base class

} // end namespace rocrand_device

#endif // ROCRAND_THREEFRY2_IMPL_H_
