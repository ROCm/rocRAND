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

#ifndef ROCRAND_LFSR113_H_
#define ROCRAND_LFSR113_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include "rocrand/rocrand_common.h"

//#define ROCRAND_LFSR113_DEFAULT_SEED {3, 9, 17, 129}
#define ROCRAND_LFSR113_DEFAULT_SEED 0ULL

namespace rocrand_device {
namespace detail {

} // end namespace detail

class lfsr113_engine
{
public:
    struct lfsr113_state
    {
        unsigned int z[4];
        unsigned int stream[4];
        unsigned int substream[4];
        unsigned int curr_stream[4] = {987654321, 987654321, 987654321, 987654321};
    };

    FQUALIFIERS
    lfsr113_engine(const unsigned int subsequence,
                   const unsigned int offset)
    {
        m_state.stream[0] = m_state.curr_stream[0];
        m_state.stream[1] = m_state.curr_stream[1];
        m_state.stream[2] = m_state.curr_stream[2];
        m_state.stream[3] = m_state.curr_stream[3];

        resetStartStream();

        int z, b;

        z = m_state.curr_stream[0] & -2;
        b = (z << 6) ^ z;

        z = (z) ^ (z << 2) ^ (z << 3) ^ (z << 10) ^ (z << 13) ^
            (z << 16) ^ (z << 19) ^ (z << 22) ^ (z << 25) ^
            (z << 27) ^ (z << 28) ^
            ((b >> 3) & 0x1FFFFFFF) ^
            ((b >> 4) & 0x0FFFFFFF) ^
            ((b >> 6) & 0x03FFFFFF) ^
            ((b >> 9) & 0x007FFFFF) ^
            ((b >> 12) & 0x000FFFFF) ^
            ((b >> 15) & 0x0001FFFF) ^
            ((b >> 18) & 0x00003FFF) ^
            ((b >> 21) & 0x000007FF);
        m_state.curr_stream[0] = z;

        z = m_state.curr_stream[1] & -8;
        b = (z << 2) ^ z;
        z = ((b >> 13) & 0x0007FFFF) ^ (z << 16);
        m_state.curr_stream[1] = z;

        z = m_state.curr_stream[2] & -16;
        b = (z << 13) ^ z;
        z = (z << 2) ^ (z << 4) ^ (z << 10) ^ (z << 12) ^ (z << 13) ^
            (z << 17) ^ (z << 25) ^
            ((b >> 3) & 0x1FFFFFFF) ^
            ((b >> 11) & 0x001FFFFF) ^
            ((b >> 15) & 0x0001FFFF) ^
            ((b >> 16) & 0x0000FFFF) ^
            ((b >> 24) & 0x000000FF);
        m_state.curr_stream[2] = z;

        z = m_state.curr_stream[3] & -128;
        b = (z << 3) ^ z;
        z = (z << 9) ^ (z << 10) ^ (z << 11) ^ (z << 14) ^ (z << 16) ^
            (z << 18) ^ (z << 23) ^ (z << 24) ^
            ((b >> 1) & 0x7FFFFFFF) ^
            ((b >> 2) & 0x3FFFFFFF) ^
            ((b >> 7) & 0x01FFFFFF) ^
            ((b >> 9) & 0x007FFFFF) ^
            ((b >> 11) & 0x001FFFFF) ^
            ((b >> 14) & 0x0003FFFF) ^
            ((b >> 15) & 0x0001FFFF) ^
            ((b >> 16) & 0x0000FFFF) ^
            ((b >> 23) & 0x000001FF) ^
            ((b >> 24) & 0x000000FF);
        m_state.curr_stream[3] = z;

        discard(subsequence);
        discard(offset);
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
        unsigned long b;

        b = (((m_state.z[0] << 6) ^ m_state.z[0]) >> 13);
        m_state.z[0] = (((m_state.z[0] & 4294967294U) << 18) ^ b);

        b = (((m_state.z[1] << 2) ^ m_state.z[1]) >> 27);
        m_state.z[1] = (((m_state.z[1] & 4294967288U) << 2) ^ b);

        b = (((m_state.z[2] << 13) ^ m_state.z[2]) >> 21);
        m_state.z[2] = (((m_state.z[2] & 4294967280U) << 7) ^ b);

        b = (((m_state.z[3] << 3) ^ m_state.z[3]) >> 12);
        m_state.z[3] = (((m_state.z[3] & 4294967168U) << 13) ^ b);

        return (m_state.z[0] ^ m_state.z[1] ^ m_state.z[2] ^ m_state.z[3]);
    }

    FQUALIFIERS
    void resetStartStream()
    {
        m_state.substream[0] = m_state.stream[0];
        m_state.substream[1] = m_state.stream[1];
        m_state.substream[2] = m_state.stream[2];
        m_state.substream[3] = m_state.stream[3];

        resetStartSubstream();
    }

    FQUALIFIERS
    void resetStartSubstream()
    {
        m_state.z[0] = m_state.substream[0];
        m_state.z[1] = m_state.substream[1];
        m_state.z[2] = m_state.substream[2];
        m_state.z[3] = m_state.substream[3];
    }

    FQUALIFIERS
    void resetNextSubstream() // Advance state
    {
        int z, b;

        z = m_state.substream[0] & -2;
        b = (z) ^ (z << 3) ^ (z << 4) ^ (z << 6) ^ (z << 7) ^
            (z << 8) ^ (z << 10) ^ (z << 11) ^ (z << 13) ^ (z << 14) ^
            (z << 16) ^ (z << 17) ^ (z << 18) ^ (z << 22) ^
            (z << 24) ^ (z << 25) ^ (z << 26) ^ (z << 28) ^ (z << 30);
        z ^= ((b >> 1) & 0x7FFFFFFF) ^ ((b >> 3) & 0x1FFFFFFF) ^
             ((b >> 5) & 0x07FFFFFF) ^ ((b >> 6) & 0x03FFFFFF) ^
             ((b >> 7) & 0x01FFFFFF) ^ ((b >> 9) & 0x007FFFFF) ^
             ((b >> 13) & 0x0007FFFF) ^ ((b >> 14) & 0x0003FFFF) ^
             ((b >> 15) & 0x0001FFFF) ^ ((b >> 17) & 0x00007FFF) ^
             ((b >> 18) & 0x00003FFF) ^ ((b >> 20) & 0x00000FFF) ^
             ((b >> 21) & 0x000007FF) ^ ((b >> 23) & 0x000001FF) ^
             ((b >> 24) & 0x000000FF) ^ ((b >> 25) & 0x0000007F) ^
             ((b >> 26) & 0x0000003F) ^ ((b >> 27) & 0x0000001F) ^
             ((b >> 30) & 0x00000003);
        m_state.substream[0] = z;

        z = m_state.substream[1] & -8;
        b = z ^ (z << 1);
        b ^= (b << 2);
        b ^= (b << 4);
        b ^= (b << 8);

        b <<= 8;
        b ^= (z << 22) ^ (z << 25) ^ (z << 27);
        if ((z & 0x80000000) != 0) b ^= 0xABFFF000;
        if ((z & 0x40000000) != 0) b ^= 0x55FFF800;
        z = b ^ ((z >> 7) & 0x01FFFFFF) ^
                ((z >> 20) & 0x00000FFF) ^
                ((z >> 21) & 0x000007FF);
        m_state.substream[1] = z;

        z = m_state.substream[2] & -16;
        b = (z << 13) ^ z;
        z = ((b >> 3) & 0x1FFFFFFF) ^ ((b >> 17) & 0x00007FFF) ^
            (z << 10) ^ (z << 11) ^ (z << 25);
        m_state.substream[2] = z;

        z = m_state.substream[3] & -128;
        b = (z << 3) ^ z;
        z = (z << 14) ^ (z << 16) ^ (z << 20) ^
            ((b >> 5) & 0x07FFFFFF) ^
            ((b >> 9) & 0x007FFFFF) ^
            ((b >> 11) & 0x001FFFFF);
        m_state.substream[3] = z;

        resetStartSubstream();
    }

protected:
    FQUALIFIERS
    void discard_state(unsigned int offset)
    {
        for (unsigned int i = 0; i < offset; i++)
            this->next();
    }

    FQUALIFIERS
    void discard_state()
    {
        this->next();
    }

protected:
    lfsr113_state m_state;

}; // lfsr113_engine class

} // end namespace rocrand_device

typedef rocrand_device::lfsr113_engine rocrand_state_lfsr113;

FQUALIFIERS
void rocrand_init(/*const unsigned int seed[4],*/
                  const unsigned int subsequence,
                  const unsigned int offset,
                  rocrand_state_lfsr113 * state) 
{
    *state = rocrand_state_lfsr113(subsequence, offset);
}

FQUALIFIERS
unsigned int rocrand(rocrand_state_lfsr113 * state)
{
    return state->next();
}

#endif // ROCRAND_LFSR113_H_