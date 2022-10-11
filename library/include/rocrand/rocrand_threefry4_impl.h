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

#ifndef ROCRAND_THREEFRY4_IMPL_H_
#define ROCRAND_THREEFRY4_IMPL_H_

#ifndef FQUALIFIERS
    #define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include "rocrand/rocrand_threefry_common.h"
#include <rocrand/rocrand_common.h>

#ifndef THREEFRY4x32_DEFAULT_ROUNDS
    #define THREEFRY4x32_DEFAULT_ROUNDS 20
#endif

#ifndef THREEFRY4x64_DEFAULT_ROUNDS
    #define THREEFRY4x64_DEFAULT_ROUNDS 20
#endif

/* These are the R_256 constants from the Threefish reference sources
   with names changed to R_64x4... */
static constexpr __device__ int THREEFRY_ROTATION_64_4[][8] = {
    {14, 52, 23,  5, 25, 46, 58, 32},
    {16, 57, 40, 37, 33, 12, 22, 32}
};

/* Output from skein_rot_search: (srs-B128-X5000.out)
// Random seed = 1. BlockSize = 64 bits. sampleCnt =  1024. rounds =  8, minHW_or=28
// Start: Mon Aug 24 22:41:36 2009
// ...
// rMin = 0.472. #0A4B[*33] [CRC=DD1ECE0F. hw_OR=31. cnt=16384. blkSize= 128].format    */
static constexpr __device__ int THREEFRY_ROTATION_32_4[][8] = {
    {10, 11, 13, 23,  6, 17, 25, 18},
    {26, 21, 27,  5, 20, 11, 10, 20}
};

namespace rocrand_device
{

template<class value>
FQUALIFIERS int threefry_rotation_array(int indexX, int indexY);

template<>
FQUALIFIERS int threefry_rotation_array<unsigned int>(int indexX, int indexY)
{
    return THREEFRY_ROTATION_32_4[indexX][indexY];
};

template<>
FQUALIFIERS int threefry_rotation_array<unsigned long long>(int indexX, int indexY)
{
    return THREEFRY_ROTATION_64_4[indexX][indexY];
};

template<typename state_value, typename value, unsigned int Nrounds>
class threefry_engine4_base
{
public:
    struct threefry_state_4
    {
        state_value  counter;
        state_value  key;
        state_value  result;
        unsigned int substate;
    };

    /// Advances the internal state to skip \p offset numbers.
    FQUALIFIERS void discard(unsigned long long offset)
    {
        this->discard_impl(offset);
        this->m_state.result = this->threefry_rounds(m_state.counter, m_state.key);
    }

    /// Advances the internal state to skip \p subsequence subsequences,
    /// a subsequence consisting of 4 * (2 ^ b) random numbers,
    /// where b is the number of bits of the value type of the generator.
    /// In other words, this function is equivalent to calling \p discard
    /// 4 * (2 ^ b) times without using the return value, but is much faster.
    FQUALIFIERS void discard_subsequence(unsigned long long subsequence)
    {
        this->discard_subsequence_impl(subsequence);
        m_state.result = this->threefry_rounds(m_state.counter, m_state.key);
    }

    FQUALIFIERS value operator()()
    {
        return this->next();
    }

    FQUALIFIERS value next()
    {
#if defined(__HIP_PLATFORM_HCC__) || defined(__HIP_PLATFORM_AMD__)
        value ret = m_state.result.data[m_state.substate];
#else
        value ret = (&m_state.result.x)[m_state.substate];
#endif
        m_state.substate++;
        if(m_state.substate == 4)
        {
            m_state.substate = 0;
            m_state.counter  = this->bump_counter(m_state.counter);
            m_state.result   = this->threefry_rounds(m_state.counter, m_state.key);
        }
        return ret;
    }

    FQUALIFIERS state_value next4()
    {
        state_value ret = m_state.result;
        m_state.counter = this->bump_counter(m_state.counter);
        m_state.result  = this->threefry_rounds(m_state.counter, m_state.key);

        return this->interleave(ret, m_state.result);
    }

protected:
    FQUALIFIERS state_value threefry_rounds(state_value counter, state_value key)
    {
        state_value X;
        value       ks[4 + 1];

        static_assert(Nrounds <= 72, "72 or less only supported in threefry rounds");

        ks[4] = skein_ks_parity<value>();

        ks[0] = key.x;
        ks[1] = key.y;
        ks[2] = key.z;
        ks[3] = key.w;

        X.x = counter.x;
        X.y = counter.y;
        X.z = counter.z;
        X.w = counter.w;

        ks[4] ^= key.x;
        ks[4] ^= key.y;
        ks[4] ^= key.z;
        ks[4] ^= key.w;

        /* Insert initial key before round 0 */
        X.x += ks[0];
        X.y += ks[1];
        X.z += ks[2];
        X.w += ks[3];

        if(Nrounds > 0)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(0, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 1)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(1, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 2)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(2, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 3)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(3, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 3)
        {
            /* InjectKey(r=1) */
            X.x += ks[1];
            X.y += ks[2];
            X.z += ks[3];
            X.w += ks[4];
            X.w += 1; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 4)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(4, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 5)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(5, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 6)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(6, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 7)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(7, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 7)
        {
            /* InjectKey(r=2) */
            X.x += ks[2];
            X.y += ks[3];
            X.z += ks[4];
            X.w += ks[0];
            X.w += 2; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 8)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(0, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 9)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(1, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 10)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(2, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 11)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(3, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 11)
        {
            /* InjectKey(r=3) */
            X.x += ks[3];
            X.y += ks[4];
            X.z += ks[0];
            X.w += ks[1];
            X.w += 3; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 12)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(4, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 13)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(5, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 14)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(6, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 15)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(7, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 15)
        {
            /* InjectKey(r=1) */
            X.x += ks[4];
            X.y += ks[0];
            X.z += ks[1];
            X.w += ks[2];
            X.w += 4; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 16)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(0, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 17)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(1, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 18)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(2, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 19)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(3, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 19)
        {
            /* InjectKey(r=1) */
            X.x += ks[0];
            X.y += ks[1];
            X.z += ks[2];
            X.w += ks[3];
            X.w += 5; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 20)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(4, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 21)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(5, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 22)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(6, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 23)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(7, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 23)
        {
            /* InjectKey(r=1) */
            X.x += ks[1];
            X.y += ks[2];
            X.z += ks[3];
            X.w += ks[4];
            X.w += 6; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 24)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(0, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 25)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(1, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 26)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(2, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 27)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(3, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 27)
        {
            /* InjectKey(r=1) */
            X.x += ks[2];
            X.y += ks[3];
            X.z += ks[4];
            X.w += ks[0];
            X.w += 7; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 28)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(4, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 29)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(5, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 30)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(6, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 31)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(7, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 31)
        {
            /* InjectKey(r=1) */
            X.x += ks[3];
            X.y += ks[4];
            X.z += ks[0];
            X.w += ks[1];
            X.w += 8; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 32)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(0, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 33)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(1, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 34)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(2, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 35)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(3, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 35)
        {
            /* InjectKey(r=1) */
            X.x += ks[4];
            X.y += ks[0];
            X.z += ks[1];
            X.w += ks[2];
            X.w += 9; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 36)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(4, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 37)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(5, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 38)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(6, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 39)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(7, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 39)
        {
            /* InjectKey(r=1) */
            X.x += ks[0];
            X.y += ks[1];
            X.z += ks[2];
            X.w += ks[3];
            X.w += 10; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 40)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(0, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 41)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(1, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 42)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(2, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 43)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(3, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 43)
        {
            /* InjectKey(r=1) */
            X.x += ks[1];
            X.y += ks[2];
            X.z += ks[3];
            X.w += ks[4];
            X.w += 11; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 44)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(4, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 45)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(5, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 46)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(6, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 47)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(7, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 47)
        {
            /* InjectKey(r=1) */
            X.x += ks[2];
            X.y += ks[3];
            X.z += ks[4];
            X.w += ks[0];
            X.w += 12; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 48)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(0, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 49)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(1, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 50)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(2, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 51)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(3, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 51)
        {
            /* InjectKey(r=1) */
            X.x += ks[3];
            X.y += ks[4];
            X.z += ks[0];
            X.w += ks[1];
            X.w += 13; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 52)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(4, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 53)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(5, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 54)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(6, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 55)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(7, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 55)
        {
            /* InjectKey(r=1) */
            X.x += ks[4];
            X.y += ks[0];
            X.z += ks[1];
            X.w += ks[2];
            X.w += 14; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 56)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(0, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 57)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(1, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 58)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(2, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 59)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(3, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 59)
        {
            /* InjectKey(r=1) */
            X.x += ks[0];
            X.y += ks[1];
            X.z += ks[2];
            X.w += ks[3];
            X.w += 15; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 60)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(4, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 61)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(5, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 62)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(6, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 63)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(7, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 63)
        {
            /* InjectKey(r=1) */
            X.x += ks[1];
            X.y += ks[2];
            X.z += ks[3];
            X.w += ks[4];
            X.w += 16; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 64)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(0, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(0, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 65)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(1, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(1, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 66)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(2, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(2, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 67)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(3, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(3, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 67)
        {
            /* InjectKey(r=1) */
            X.x += ks[2];
            X.y += ks[3];
            X.z += ks[4];
            X.w += ks[0];
            X.w += 17; /* X.v[WCNT4-1] += r  */
        }

        if(Nrounds > 68)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(4, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(4, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 69)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(5, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(5, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 70)
        {
            X.x += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(6, 0));
            X.y ^= X.x;
            X.z += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(6, 1));
            X.w ^= X.z;
        }
        if(Nrounds > 71)
        {
            X.x += X.w;
            X.w = rotl<value>(X.w, threefry_rotation_array<value>(7, 0));
            X.w ^= X.x;
            X.z += X.y;
            X.y = rotl<value>(X.y, threefry_rotation_array<value>(7, 1));
            X.y ^= X.z;
        }
        if(Nrounds > 71)
        {
            /* InjectKey(r=1) */
            X.x += ks[3];
            X.y += ks[4];
            X.z += ks[0];
            X.w += ks[1];
            X.w += 18; /* X.v[WCNT4-1] += r  */
        }

        return X;
    }

    /// Advances the internal state to skip \p offset numbers.
    /// Does not calculate new values (or update <tt>m_state.result</tt>).
    FQUALIFIERS void discard_impl(unsigned long long offset)
    {
        // Adjust offset for subset
        m_state.substate += offset & 3;
        unsigned long long counter_offset = offset / 4;
        counter_offset += m_state.substate < 4 ? 0 : 1;
        m_state.substate += m_state.substate < 4 ? 0 : -4;
        // Discard states
        this->discard_state(counter_offset);
    }

    /// Does not calculate new values (or update <tt>m_state.result</tt>).
    FQUALIFIERS void discard_subsequence_impl(unsigned long long subsequence)
    {
        value lo, hi;
        ::rocrand_device::detail::split_ull(lo, hi, subsequence);

        value old_counter = m_state.counter.z;
        m_state.counter.z += lo;
        m_state.counter.w += hi + (m_state.counter.z < old_counter ? 1 : 0);
    }

    /// Advances the internal state by \p offset times.
    /// Does not calculate new values (or update <tt>m_state.result</tt>).
    FQUALIFIERS void discard_state(unsigned long long offset)
    {
        value lo, hi;
        ::rocrand_device::detail::split_ull(lo, hi, offset);

        state_value old_counter = m_state.counter;
        m_state.counter.x += lo;
        m_state.counter.y += hi + (m_state.counter.x < old_counter.x ? 1 : 0);
        m_state.counter.z += (m_state.counter.y < old_counter.y ? 1 : 0);
        m_state.counter.w += (m_state.counter.z < old_counter.z ? 1 : 0);
    }

    FQUALIFIERS static state_value bump_counter(state_value counter)
    {
        counter.x++;
        value add = counter.x == 0 ? 1 : 0;
        counter.y += add;
        add = counter.y == 0 ? add : 0;
        counter.z += add;
        add = counter.z == 0 ? add : 0;
        counter.w += add;
        return counter;
    }

    FQUALIFIERS state_value interleave(const state_value prev, const state_value next) const
    {
        switch(m_state.substate)
        {
            case 0: return prev;
            case 1: return state_value{prev.y, prev.z, prev.w, next.x};
            case 2: return state_value{prev.z, prev.w, next.x, next.y};
            case 3: return state_value{prev.w, next.x, next.y, next.z};
        }
        __builtin_unreachable();
    }

protected:
    threefry_state_4 m_state;
}; // threefry_engine4_base class

} // end namespace rocrand_device

#endif // ROCRAND_THREEFRY4_IMPL_H_
