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

#ifndef ROCRAND_LFSR113_H_
#define ROCRAND_LFSR113_H_

#ifndef FQUALIFIERS
    #define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include "rocrand/rocrand_common.h"

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */
/// \def ROCRAND_LFSR113_DEFAULT_SEED_X
/// \brief Default X seed for LFSR113 PRNG.
#define ROCRAND_LFSR113_DEFAULT_SEED_X 2
/// \def ROCRAND_LFSR113_DEFAULT_SEED_Y
/// \brief Default Y seed for LFSR113 PRNG.
#define ROCRAND_LFSR113_DEFAULT_SEED_Y 8
/// \def ROCRAND_LFSR113_DEFAULT_SEED_Z
/// \brief Default Z seed for LFSR113 PRNG.
#define ROCRAND_LFSR113_DEFAULT_SEED_Z 16
/// \def ROCRAND_LFSR113_DEFAULT_SEED_W
/// \brief Default W seed for LFSR113 PRNG.
#define ROCRAND_LFSR113_DEFAULT_SEED_W 128
/** @} */ // end of group rocranddevice

namespace rocrand_device
{

class lfsr113_engine
{
public:
    struct lfsr113_state
    {
        uint4 z;
        uint4 subsequence;
    };

    /// Initializes the internal state of the PRNG using
    /// seed value \p seed, goes to \p subsequence -th subsequence
    ///
    /// A subsequence is 2^55 numbers long.
    FQUALIFIERS
    lfsr113_engine(const uint4        seed        = {ROCRAND_LFSR113_DEFAULT_SEED_X,
                                                     ROCRAND_LFSR113_DEFAULT_SEED_Y,
                                                     ROCRAND_LFSR113_DEFAULT_SEED_Z,
                                                     ROCRAND_LFSR113_DEFAULT_SEED_W},
                   const unsigned int subsequence = 0)
    {
        this->seed(seed, subsequence);
    }

    /// Reinitializes the internal state of the PRNG using new
    /// seed value \p seed_value, skips \p subsequence subsequences.
    ///
    /// A subsequence is 2^55 numbers long.
    FQUALIFIERS
    void seed(uint4 seed_value, const unsigned long long subsequence)
    {
        m_state.subsequence = seed_value;

        reset_start_subsequence();
        discard_subsequence(subsequence);
    }

    /// Advances the internal state to skip one number.
    FQUALIFIERS
    void discard()
    {
        discard_state();
    }

    /// Advances the internal state to skip \p subsequence subsequences.
    /// A subsequence is 2^55 numbers long.
    FQUALIFIERS
    void discard_subsequence(unsigned int subsequence)
    {
        for(unsigned int i = 0; i < subsequence; i++)
        {
            reset_next_subsequence();
        }
    }

    FQUALIFIERS
    unsigned int operator()()
    {
        return next();
    }

    FQUALIFIERS
    unsigned int next()
    {
        unsigned int b;

        b           = (((m_state.z.x << 6) ^ m_state.z.x) >> 13);
        m_state.z.x = (((m_state.z.x & 4294967294U) << 18) ^ b);

        b           = (((m_state.z.y << 2) ^ m_state.z.y) >> 27);
        m_state.z.y = (((m_state.z.y & 4294967288U) << 2) ^ b);

        b           = (((m_state.z.z << 13) ^ m_state.z.z) >> 21);
        m_state.z.z = (((m_state.z.z & 4294967280U) << 7) ^ b);

        b           = (((m_state.z.w << 3) ^ m_state.z.w) >> 12);
        m_state.z.w = (((m_state.z.w & 4294967168U) << 13) ^ b);

        return (m_state.z.x ^ m_state.z.y ^ m_state.z.z ^ m_state.z.w);
    }

protected:
    /// Resets the state to the start of the current subsequence.
    FQUALIFIERS
    void reset_start_subsequence()
    {
        m_state.z.x = m_state.subsequence.x;
        m_state.z.y = m_state.subsequence.y;
        m_state.z.z = m_state.subsequence.z;
        m_state.z.w = m_state.subsequence.w;
    }

    /// Advances the subsequence by one and sets the state to the start of that subsequence.
    FQUALIFIERS
    void reset_next_subsequence()
    {
        /* The following operations make the jump ahead with
    	2 ^ 55 iterations for every component of the generator.
    	The internal state after the jump, however, is slightly different
    	from 2 ^ 55 iterations since it ignores the state in
    	which are found the first bits of each components,
    	since they are ignored in the recurrence.The state becomes
    	identical to what one would with normal iterations
    	after a call nextValue().*/
        int z, b;

        z = m_state.subsequence.x & 0xFFFFFFFE;
        b = (z << 6) ^ z;

        z = (z) ^ (z << 3) ^ (z << 4) ^ (z << 6) ^ (z << 7) ^ (z << 8) ^ (z << 10) ^ (z << 11)
            ^ (z << 13) ^ (z << 14) ^ (z << 16) ^ (z << 17) ^ (z << 18) ^ (z << 22) ^ (z << 24)
            ^ (z << 25) ^ (z << 26) ^ (z << 28) ^ (z << 30);

        z ^= ((b >> 1) & 0x7FFFFFFF) ^ ((b >> 3) & 0x1FFFFFFF) ^ ((b >> 5) & 0x07FFFFFF)
             ^ ((b >> 6) & 0x03FFFFFF) ^ ((b >> 7) & 0x01FFFFFF) ^ ((b >> 9) & 0x007FFFFF)
             ^ ((b >> 13) & 0x0007FFFF) ^ ((b >> 14) & 0x0003FFFF) ^ ((b >> 15) & 0x0001FFFF)
             ^ ((b >> 17) & 0x00007FFF) ^ ((b >> 18) & 0x00003FFF) ^ ((b >> 20) & 0x00000FFF)
             ^ ((b >> 21) & 0x000007FF) ^ ((b >> 23) & 0x000001FF) ^ ((b >> 24) & 0x000000FF)
             ^ ((b >> 25) & 0x0000007F) ^ ((b >> 26) & 0x0000003F) ^ ((b >> 27) & 0x0000001F)
             ^ ((b >> 30) & 0x00000003);
        m_state.subsequence.x = z;

        z = m_state.subsequence.y & 0xFFFFFFF8;
        b = z ^ (z << 1);
        b ^= (b << 2);
        b ^= (b << 4);
        b ^= (b << 8);

        b <<= 8;
        b ^= (z << 22) ^ (z << 25) ^ (z << 27);
        if((z & 0x80000000) != 0)
            b ^= 0xABFFF000;
        if((z & 0x40000000) != 0)
            b ^= 0x55FFF800;
        z = b ^ ((z >> 7) & 0x01FFFFFF) ^ ((z >> 20) & 0x00000FFF) ^ ((z >> 21) & 0x000007FF);
        m_state.subsequence.y = z;

        z = m_state.subsequence.z & 0xFFFFFFF0;
        b = (z << 13) ^ z;
        z = ((b >> 3) & 0x1FFFFFFF) ^ ((b >> 17) & 0x00007FFF) ^ (z << 10) ^ (z << 11) ^ (z << 25);
        m_state.subsequence.z = z;

        z = m_state.subsequence.w & 0xFFFFFF80;
        b = (z << 3) ^ z;
        z = (z << 14) ^ (z << 16) ^ (z << 20) ^ ((b >> 5) & 0x07FFFFFF) ^ ((b >> 9) & 0x007FFFFF)
            ^ ((b >> 11) & 0x001FFFFF);
        m_state.subsequence.w = z;

        reset_start_subsequence();
    }

    // Advances the internal state to the next state.
    FQUALIFIERS
    void discard_state()
    {
        this->next();
    }

protected:
    lfsr113_state m_state;

}; // lfsr113_engine class

} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/// \cond ROCRAND_KERNEL_DOCS_TYPEDEFS
typedef rocrand_device::lfsr113_engine rocrand_state_lfsr113;
/// \endcond

/**
 * \brief Initializes LFSR113 state.
 *
 * Initializes the LFSR113 generator \p state with the given
 * \p seed, \p subsequence, and \p offset.
 *
 * \param seed - Value to use as a seed
 * \param subsequence - Subsequence to start at
 * \param state - Pointer to state to initialize
 */
FQUALIFIERS
void rocrand_init(const uint4 seed, const unsigned int subsequence, rocrand_state_lfsr113* state)
{
    *state = rocrand_state_lfsr113(seed, subsequence);
}

/**
 * \brief Returns uniformly distributed random <tt>unsigned int</tt> value
 * from [0; 2^32 - 1] range.
 *
 * Generates and returns uniformly distributed random <tt>unsigned int</tt>
 * value from [0; 2^32 - 1] range using LFSR113 generator in \p state.
 * State is incremented by one position.
 *
 * \param state - Pointer to a state to use
 *
 * \return Pseudorandom value (32-bit) as an <tt>unsigned int</tt>
 */
FQUALIFIERS
unsigned int rocrand(rocrand_state_lfsr113* state)
{
    return state->next();
}

/** @} */ // end of group rocranddevice

#endif // ROCRAND_LFSR113_H_
