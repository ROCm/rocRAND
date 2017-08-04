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

#ifndef ROCRAND_SOBOL32_H_
#define ROCRAND_SOBOL32_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS_

#include "rocrand_common.h"

// S. Joe and F. Y. Kuo, Remark on Algorithm 659: Implementing Sobol's quasirandom
// sequence generator, 2003
// http://doi.acm.org/10.1145/641876.641879

namespace rocrand_device {

class sobol32_engine
{
public:
    struct sobol32_state
    {
        unsigned int i, d;
        unsigned int vectors[32];

        FQUALIFIERS
        ~sobol32_state() { }
    };

    FQUALIFIERS
    sobol32_engine()
    {
        m_state.d = 0;
        m_state.i = (unsigned int) -1;
    }

    FQUALIFIERS
    sobol32_engine(const unsigned int * vectors,
                   const unsigned int offset)
    {
        this->seed(vectors, offset);
    }

    FQUALIFIERS
    ~sobol32_engine() { }

    /// Reinitializes the internal state of the QRNG using new
    /// direction vector \p vectors, and \p offset random numbers.
    FQUALIFIERS
    void seed(const unsigned int * vectors,
              const unsigned int offset)
    {
        #pragma unroll
        for(int i = 0; i < 32; i++)
        {
            m_state.vectors[i] = vectors[i];
        }
        this->restart(offset);
    }

    /// Advances the internal state to skip \p offset numbers.
    FQUALIFIERS
    void discard(unsigned int offset)
    {
        this->discard_impl(offset);
    }

    FQUALIFIERS
    void discard()
    {
        this->discard_state();
    }

    FQUALIFIERS
    void restart(const unsigned int offset)
    {
        m_state.d = 0;
        m_state.i = (unsigned int) -1;
        this->discard_impl(offset);
    }

    FQUALIFIERS
    unsigned int operator()()
    {
        return this->next();
    }

    FQUALIFIERS
    unsigned int next()
    {
        unsigned int p = m_state.d;
        discard_state();
        return p;
    }

protected:
    // Advances the internal state to skip \p offset numbers.
    // DOES NOT CALCULATE NEW UINT
    FQUALIFIERS
    void discard_impl(unsigned int offset)
    {
        discard_state(offset);
    }

    // Advances the internal state by offset times.
    // DOES NOT CALCULATE NEW UINT
    FQUALIFIERS
    void discard_state(unsigned int offset)
    {
        unsigned int dx = sobol_bits(m_state.i, offset);
        unsigned int dy = m_state.d;
        unsigned int c  = 0;

        while(dx)
        {
            if (dx & 1)
                dy ^= m_state.vectors[c];
            dx >>= 1;
            c += 1;
        }

        m_state.i -= offset;
        m_state.d = dy;
    }

    // Advances the internal state to the next state
    // DOES NOT CALCULATE NEW UINT
    FQUALIFIERS
    void discard_state()
    {
        unsigned int c = ctz(m_state.i);
        m_state.d ^= m_state.vectors[c];
        m_state.i--;
    }

private:
    FQUALIFIERS
    unsigned int sobol_bits(unsigned int x, unsigned int offset)
    {
        unsigned int i = ~x;
        unsigned int n = i + offset;
        unsigned int a = i ^ (i >> 1);
        unsigned int b = n ^ (n >> 1);
        unsigned int d = a ^ b;

        return d;
    }

    FQUALIFIERS
    unsigned int ctz(unsigned int x)
    {
        #if defined(__HIP_DEVICE_COMPILE__)
        unsigned int z = __ffs(x);
        return z ? z - 1 : 0;
        #else
        if(x == 0)
            return 0;
        unsigned int y = ~x;
        unsigned int z = 1;
        while(y & 1)
        {
            z++;
            y >>= 1;
        }
        return z - 1;
        #endif
    }

protected:
    // State
    sobol32_state m_state;

}; // sobol32_engine class

} // end namespace rocrand_device

typedef rocrand_device::sobol32_engine rocrand_state_sobol32;

/**
 * \brief Initialize SOBOL32 state.
 *
 * Initialize SOBOL32 state in \p state with the given \p vectors and \p offset.
 *
 * \param vectors - Direction vectors
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
FQUALIFIERS
void rocrand_init(const unsigned int * vectors,
                  const unsigned int offset,
                  rocrand_state_sobol32 * state)
{
    *state = rocrand_state_sobol32(vectors, offset);
}

/**
 * \brief Return pseudorandom value (32-bit) from SOBOL32 generator.
 *
 * Return pseudorandom value (32-bit) from the SOBOL32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return pseudorandom value (32-bit) as an unsigned int
 */
FQUALIFIERS
unsigned int rocrand(rocrand_state_sobol32 * state)
{
    return state->next();
}

/**
 * \brief Update SOBOL32 state to skip ahead by \p offset elements.
 *
 * Update the SOBOL32 state in \p state to skip ahead by \p offset elements.
 *
 * \param offset - Number of elements to skip
 * \param state - Pointer to state to update
 */
FQUALIFIERS
void skipahead(unsigned long long offset, rocrand_state_sobol32 * state)
{
    return state->discard(offset);
}

#endif // ROCRAND_SOBOL32_H_
