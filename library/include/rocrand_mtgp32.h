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

#ifndef ROCRAND_MTGP32_H_
#define ROCRAND_MTGP32_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS_

#include "rocrand_common.h"

#define MTGPDC_MEXP 11213
#define MTGPDC_N 351
#define MTGPDC_FLOOR_2P 256
#define MTGPDC_CEIL_2P 512
#define MTGP_TN MTGPDC_FLOOR_2P
#define MTGP_LS (MTGP_TN * 3)
#define MTGP_BN_MAX 200
#define MTGP_TS 16
#define MTGP_STATE_SIZE 1024
#define MTGP_STATE_MASK 1023


namespace rocrand_device {

class mtgp32_engine
{
public:
    struct mtgp32_param
    {
        unsigned int pos_tbl;
        unsigned int param_tbl[MTGP_TS];
        unsigned int temper_tbl[MTGP_TS];
        unsigned int single_temper_tbl[MTGP_TS];
        unsigned int sh1_tbl;
        unsigned int sh2_tbl;
        unsigned int mask;
        
        FQUALIFIERS
        ~mtgp32_param() { }
    };
    
    struct mtgp32_state
    {
        unsigned int status[MTGP_STATE_SIZE];
        int offset;
        int id;
        
        FQUALIFIERS
        ~mtgp32_state() { }
    };

    FQUALIFIERS
    mtgp32_engine()
    {

    }

    FQUALIFIERS
    mtgp32_engine(const unsigned int seed,
                  const unsigned int offset)
    {

    }

    FQUALIFIERS
    ~mtgp32_engine() { }

    /// Reinitializes the internal state of the PRNG using new
    /// direction vector \p vectors, and \p offset random numbers.
    FQUALIFIERS
    void seed(const unsigned int seed,
              const unsigned int offset)
    {

    }

    FQUALIFIERS
    void restart(const unsigned int offset)
    {

    }

    FQUALIFIERS
    unsigned int operator()()
    {
        return this->next();
    }

    FQUALIFIERS
    unsigned int next()
    {
        unsigned int t = (hipBlockIdx_z * hipBlockIdx_y * hipThreadIdx_z) + 
                         (hipBlockIdx_x * hipThreadIdx_y) + hipThreadIdx_x;
        unsigned int d = hipBlockIdx_z * hipBlockIdx_y * hipBlockIdx_x;
        int pos = m_param.pos_tbl;
        unsigned int r;
        unsigned int o;
        
        r = para_rec(m_state.status[(t + m_state.offset) & MTGP_STATE_MASK],
                     m_state.status[(t + m_state.offset + 1) & MTGP_STATE_MASK],
                     m_state.status[(t + m_state.offset + pos) & MTGP_STATE_MASK]);
        m_state.status[(t + m_state.offset + MTGPDC_N) & MTGP_STATE_MASK] = r;
        
        o = temper(r, m_state.status[(t + m_state.offset + pos -1) & MTGP_STATE_MASK]);
        #if defined(__HIP_DEVICE_COMPILE__)
        __syncthreads();
        #endif
        if (t == 0)
            m_state.offset = (m_state.offset + d) & MTGP_STATE_MASK;
        #if defined(__HIP_DEVICE_COMPILE__)
        __syncthreads();
        #endif
        return o;
    }

private:
    FQUALIFIERS 
    unsigned int recursion(unsigned int X1, unsigned int X2, unsigned int Y) 
    {
        unsigned int X = (X1 & mask[0]) ^ X2;
        unsigned int MAT;

        X ^= X << m_param.sh1_tbl;
        Y = X ^ (Y >> m_param.sh2_tbl);
        MAT = m_param.param_tbl[Y & 0x0f];
        return Y ^ MAT;
    }
    
    FQUALIFIERS
    unsigned int temper(unsigned int V, unsigned int T) 
    {
        unsigned int MAT;

        T ^= T >> 16;
        T ^= T >> 8;
        MAT = m_param.temper_tbl[T & 0x0f];
        return V ^ MAT;
    }
    
protected:
    // State
    mtgp32_state m_state;
    mtgp32_param m_param;

}; // mtgp32_engine class

} // end namespace rocrand_device

typedef rocrand_device::mtgp32_engine rocrand_state_mtgp32;

/**
 * \brief Initialize MTGP32 state.
 *
 * Initialize MTGP32 state in \p state with the given \p vectors and \p offset.
 *
 * \param vectors - Direction vectors
 * \param offset - Absolute offset into sequence
 * \param state - Pointer to state to initialize
 */
FQUALIFIERS
void rocrand_init(const unsigned int seed,
                  const unsigned int offset,
                  rocrand_state_mtgp32 * state)
{
    //*state = rocrand_state_mtgp32(vectors, offset);
}

/**
 * \brief Return pseudorandom value (32-bit) from MTGP32 generator.
 *
 * Return pseudorandom value (32-bit) from the MTGP32 generator in \p state,
 * increment position of generator by one.
 *
 * \param state - Pointer to state to update
 *
 * \return pseudorandom value (32-bit) as an unsigned int
 */
FQUALIFIERS
unsigned int rocrand(rocrand_state_mtgp32 * state)
{
    return m_state.next();
}

#endif // ROCRAND_MTGP32_H_
