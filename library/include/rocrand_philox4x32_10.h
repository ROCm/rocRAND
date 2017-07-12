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

#ifndef ROCRAND_PHILOX4X32_10_H_
#define ROCRAND_PHILOX4X32_10_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS_

#include "rocrand.h"
#include "rocrand_common.h"

namespace detail {

struct philox4x32_10_state
{
    uint4 counter;
    uint4 result;
    uint2 key;
    unsigned int substate;

    // The Box–Muller transform requires two inputs to convert uniformly
    // distributed real values [0; 1] to normally distributed real values
    // (with mean = 0, and stddev = 1). Often user wants only one
    // normally distributed number, to save performance and random
    // numbers the 2nd value is saved for future requests.
    unsigned int boxmuller_float_state; // is there a float in boxmuller_float
    unsigned int boxmuller_double_state; // is there a double in boxmuller_double
    float boxmuller_float; // normally distributed float
    double boxmuller_double; // normally distributed double
};

namespace philox4x32_10 {

// Constants from Random123
// See https://www.deshawresearch.com/resources_random123.html
const unsigned int PHILOX_M4x32_0 = 0xD2511F53;
const unsigned int PHILOX_M4x32_1 = 0xCD9E8D57;
const unsigned int PHILOX_W32_0   = 0x9E3779B9;
const unsigned int PHILOX_W32_1   = 0xBB67AE85;

// HCC
#ifdef __HIP_DEVICE_COMPILE__
__forceinline__ __device__
unsigned int mulhilo32(unsigned int x, unsigned int y, unsigned int& z)
{
    z = __umulhi(x, y);
    return x * y;
}
#else
inline __host__
unsigned int mulhilo32(unsigned int x, unsigned int y, unsigned int& z)
{
    unsigned long long xy =
        static_cast<unsigned long long>(x) * static_cast<unsigned long long>(y);
    z = xy >> 32;
    return static_cast<unsigned int>(xy);
}
#endif

FQUALIFIERS
void set_seed(philox4x32_10_state * state, unsigned long long seed)
{
    state->key.x = static_cast<unsigned int>(seed);
    state->key.y = static_cast<unsigned int>(seed >> 32);
    state->counter = {0, 0, 0, 0};
    state->result  = {0, 0, 0, 0};
    state->substate = 0;
    state->boxmuller_float_state = 0;
    state->boxmuller_double_state = 0;
}

FQUALIFIERS
void discard(philox4x32_10_state * state, unsigned long long n)
{
    unsigned int lo = static_cast<unsigned int>(n);
    unsigned int hi = static_cast<unsigned int>(n >> 32);

    uint4 temp = state->counter;
    state->counter.x += lo;
    state->counter.y += hi + (state->counter.x < temp.x ? 1 : 0);
    state->counter.z += (state->counter.y < temp.y ? 1 : 0);
    state->counter.w += (state->counter.z < temp.z ? 1 : 0);
}

FQUALIFIERS
void discard(philox4x32_10_state * state)
{
    state->counter.x++;
    uint add = state->counter.x == 0 ? 1 : 0;
    state->counter.y += add; add = state->counter.y == 0 ? add : 0;
    state->counter.z += add; add = state->counter.z == 0 ? add : 0;
    state->counter.w += add;
}

FQUALIFIERS
void discard_subsequence(philox4x32_10_state * state,
                         unsigned long long subsequence)
{
    unsigned int lo = static_cast<unsigned int>(subsequence);
    unsigned int hi = static_cast<unsigned int>(subsequence >> 32);

    unsigned int temp = state->counter.z;
    state->counter.z += lo;
    state->counter.w += hi + (state->counter.z < temp ? 1 : 0);
}

// Single Philox4x32 round
FQUALIFIERS
uint4 single_round(uint4 counter, uint2 key)
{
    // Source: Random123
    unsigned int hi0;
    unsigned int hi1;
    unsigned int lo0 = mulhilo32(PHILOX_M4x32_0, counter.x, hi0);
    unsigned int lo1 = mulhilo32(PHILOX_M4x32_1, counter.z, hi1);
    return uint4 {
        hi1 ^ counter.y ^ key.x,
        lo1,
        hi0 ^ counter.w ^ key.y,
        lo0
    };
}

FQUALIFIERS
uint2 bumpkey(uint2 key)
{
    key.x += PHILOX_W32_0;
    key.y += PHILOX_W32_1;
    return key;
}

// 10 Philox4x32 rounds
FQUALIFIERS
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

FQUALIFIERS
void init_state(philox4x32_10_state * state,
                const unsigned long long seed,
                const unsigned long long subsequence,
                const unsigned long long offset)
{
    set_seed(state, seed);
    discard_subsequence(state, subsequence);
    discard(state, offset);
    state->result = ten_rounds(state->counter, state->key);
}

FQUALIFIERS
unsigned int next(philox4x32_10_state * state)
{
    unsigned int ret = (&state->result.x)[state->substate];
    state->substate++;
    if(state->substate == 4)
    {
        state->substate = 0;
        discard(state);
        state->result = ten_rounds(state->counter, state->key);
    }
    return ret;
}

FQUALIFIERS
uint4 next4(philox4x32_10_state * state)
{
    uint4 ret = state->result;
    discard(state);
    state->result = ten_rounds(state->counter, state->key);
    switch(state->substate)
    {
        case 0:
            return ret;
        case 1:
            ret = { ret.y, ret.z, ret.w, state->result.x };
            // ret.x = ret.y;
            // ret.y = ret.z;
            // ret.z = ret.w;
            // ret.w = state->result.x;
            break;
        case 2:
            ret = { ret.z, ret.w, state->result.x, state->result.y };
            // ret.x = ret.z;
            // ret.y = ret.w;
            // ret.z = state->result.x;
            // ret.w = state->result.y;
            break;
        case 3:
            ret = { ret.w, state->result.x, state->result.y, state->result.z };
            // ret.x = ret.w;
            // ret.y = state->result.x;
            // ret.z = state->result.y;
            // ret.w = state->result.z;
            break;
        default:
            return ret;
    }
    return ret;
}

} // end namespace philox4x32_10
} // end namespace detail

typedef detail::philox4x32_10_state rocrand_state_philox4x32_10;

FQUALIFIERS
void rocrand_init(const unsigned long long seed,
                  const unsigned long long subsequence,
                  const unsigned long long offset,
                  rocrand_state_philox4x32_10 * state)
{
    detail::philox4x32_10::init_state(state, seed, subsequence, offset);
}

FQUALIFIERS
unsigned int rocrand(rocrand_state_philox4x32_10 * state)
{
    return detail::philox4x32_10::next(state);
}

FQUALIFIERS
uint4 rocrand4(rocrand_state_philox4x32_10 * state)
{
    return detail::philox4x32_10::next4(state);
}

FQUALIFIERS
void skipahead(unsigned long long offset, rocrand_state_philox4x32_10 * state)
{
    // Adjust for substate
    state->substate += offset & 3;
    offset /= 4;
    offset += state->substate < 4 ? 0 : 1;
    state->substate += state->substate < 4 ? 0 : -4;
    return detail::philox4x32_10::discard(state, offset);
}

FQUALIFIERS
void skipahead_subsequence(unsigned long long subsequence, rocrand_state_philox4x32_10 * state)
{
    return detail::philox4x32_10::discard_subsequence(state, subsequence);
}

#endif // ROCRAND_PHILOX4X32_10_H_
