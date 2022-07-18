// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_THREEFRY_H_
#define ROCRAND_THREEFRY_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include "rocrand/rocrand_common.h"

#define THREEFRY_N_WORDS 2
#define THREEFRY_KEY_LENGTH 3
#define THREEFRY_C240 0x1BD11BDAA9FC1A22
#define THREEFRY_N_ROUNDS 20
#define THREEFRY_MASK 0xffffffffffffffff
#define THREEFRY_DOUBLE_MULT 5.421010862427522e-20

static const __device__ int THREEFRY_ROTATION[] = {16, 42, 12, 31, 16, 32, 24, 21};

namespace rocrand_device {
namespace detail {

} // end namespace detail

class threefry_engine
{
public:
    struct threefry_state
    {
        ulonglong2 counter;
        ulonglong2 key;
    };

    FQUALIFIERS
    threefry_engine(const unsigned long long seed = 0,
                    const unsigned long long subsequence = 0,
                    const unsigned long long offset = 0)
    {
        this->seed(seed, subsequence, offset);
    }

    FQUALIFIERS
    void seed(const unsigned long long seed = 0,
              const unsigned long long subsequence = 0,
              const unsigned long long offset = 0)
    {
        m_state.counter.x = 0;
        m_state.counter.y = 0;

        m_state.key.x = seed;
        m_state.key.y = 0;

        this->discard(subsequence);
        this->discard(offset);
    }

    FQUALIFIERS
    void discard(unsigned int offset)
    {
        discard_state(offset);
    }

    FQUALIFIERS
    void discard()
    {
        discard_state();
    }

    FQUALIFIERS
    unsigned int operator()()
    {
        return next();
    }

    FQUALIFIERS
    unsigned int next()
    {
        ulonglong2 x = this->next2(this->m_state.counter, this->m_state.key);
        this->m_state.counter.x++;

        return x.x;
    }

    FQUALIFIERS
    ulonglong2 next2(ulonglong2 p, ulonglong2 k)
    {
        unsigned long long K[3];
        K[0] = k.x;
        K[1] = k.y;
        K[2] = THREEFRY_C240 ^ k.x ^ k.y;

        int rmod4, rdiv4;

        ulonglong2 x;
        x.x = p.x;
        x.y = p.y;

        for (int r = 0; r < THREEFRY_N_ROUNDS; r++) {
            rmod4 = r % 4;

            if (rmod4 == 0) {
                rdiv4 = r / 4;

                x.x += K[rdiv4 % THREEFRY_KEY_LENGTH];
                x.y += K[(rdiv4 + 1) % THREEFRY_KEY_LENGTH] + rdiv4;
            }

            x = mix(x, THREEFRY_ROTATION[r % 8]);
        }

        x.x += K[(THREEFRY_N_ROUNDS / 4) % THREEFRY_KEY_LENGTH];
        x.y += K[(THREEFRY_N_ROUNDS / 4 + 1) % THREEFRY_KEY_LENGTH] + THREEFRY_N_ROUNDS / 4;

        return x;
    }

protected:
    FQUALIFIERS
    void discard_state(unsigned int offset)
    {
        for (unsigned int i = 0; i < offset; i++)
        {
            this->discard_state();
        }
    }

    FQUALIFIERS
    void discard_state()
    {
        m_state.counter.x++;
    }

    FQUALIFIERS
    unsigned long long rotl(unsigned long long x, int d)
    {
        return ((x << d) | (x >> (64 - d)));
    }

    FQUALIFIERS
    ulonglong2 mix(ulonglong2 x, int R)
    {
        x.x += x.y;
        x.y = rotl(x.y, R) ^ x.x;

        return x;
    }

protected:
    threefry_state m_state;
}; // threefry_engine class

} // end namespace rocrand_device

typedef rocrand_device::threefry_engine rocrand_state_threefry;

FQUALIFIERS
void rocrand_init(const unsigned long long seed,
                  const unsigned long long subsequence,
                  const unsigned long long offset,
                  rocrand_state_threefry * state)
{
    *state = rocrand_state_threefry(seed, subsequence, offset);
}

FQUALIFIERS
unsigned int rocrand(rocrand_state_threefry * state)
{
    return state->next();
}

#endif // ROCRAND_THREEFRY_H_
