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

static const int THREEFRY_ROTATION[] = {16, 42, 12, 31, 16, 32, 24, 21};

namespace rocrand_device {
namespace detail {

} // end namespace detail

class threefry_engine
{
public:
    struct threefry_t
    {
        unsigned int c0, c1;
    };

    FQUALIFIERS
    threefry_engine(const unsigned long long seed
                    const unsigned long long subsequence,
                    const unsigned long long offset)
    {
        
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
    threefry_t operator()(threefry_t p, threefry_t k)
    {
        return next(p, k);
    }

    FQUALIFIERS
    threefry_t next(threefry_t p, threefry_t k)
    {
        unsigned int K[3];
        K[0] = k.c0;
        K[1] = k.c1;
        k[2] = THREEFRY_C240 ^ k.c0 ^ k.c1;

        int rmod4, rdiv4;
        
        threefry_t x;
        x.c0 = p.c0;
        x.c1 = p.c1;

        for (int r = 0; r < THREEFRY_N_ROUNDS; r++) {
            rmod4 = r % 4;

            if (rmod4 == 0) {
                rdiv4 = r / 4;

                x.c0 += K[rdiv4 % THREEFRY_KEY_LENGTH];
                x.c1 += K[(rdiv4 + 1) % THREEFRY_KEY_LENGTH] + rdiv4;
            }

            x = mix(x, THREEFRY_ROTATION[r % 8]);
        }

        x.c0 += K[(THREEFRY_N_ROUNDS / 4) % THREEFRY_KEY_LENGTH];
        x.c1 += K[(THREEFRY_N_ROUNDS / 4 + 1) % THREEFRY_KEY_LENGTH] + THREEFRY_N_ROUNDS / 4;

        return x;
    }

protected:
    FQUALIFIERS
    void discard_state(unsigned int offset)
    {
        
    }

    FQUALIFIERS
    void discard_state()
    {
        
    }

    FQUALIFIERS
    unsigned int rotl(unsigned int x, int d)
    {
        return ((x << d) | (x >> (64 - d)));
    }

    FQUALIFIERS
    threefry_t mix(threefry_t x, int R)
    {
        x.c0 += x.c1;
        x.c1 = rotl(x.c1, R) ^ x.c0;

        return x; 
    }

protected:

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