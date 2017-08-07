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

#include "rocrand.h"
#include "rocrand_common.h"

#define MTGP_MEXP 11213
#define MTGP_N 351
#define MTGP_FLOOR_2P 256
#define MTGP_CEIL_2P 512
#define MTGP_TN MTGP_FLOOR_2P
#define MTGP_LS (MTGP_TN * 3)
#define MTGP_BN_MAX 200
#define MTGP_TS 16


namespace rocrand_device {
    
struct mtgp32_param
{
    unsigned int pos_tbl[MTGP_BN_MAX];
    unsigned int param_tbl[MTGP_BN_MAX][MTGP_TS];
    unsigned int temper_tbl[MTGP_BN_MAX][MTGP_TS];
    unsigned int single_temper_tbl[MTGP_BN_MAX][MTGP_TS];
    unsigned int sh1_tbl[MTGP_BN_MAX];
    unsigned int sh2_tbl[MTGP_BN_MAX];
    unsigned int mask[1];
        
    FQUALIFIERS
    ~mtgp32_param() { }
};
    
struct mtgp32_fast_param
{
    int mexp;			/**< Mersenne exponent. This is redundant. */
    int pos;			/**< pick up position. */
    int sh1;			/**< shift value 1. 0 < sh1 < 32. */
    int sh2;			/**< shift value 2. 0 < sh2 < 32. */
    unsigned int tbl[16];		/**< a small matrix. */
    unsigned int tmp_tbl[16];	/**< a small matrix for tempering. */
    unsigned int flt_tmp_tbl[16];	/**< a small matrix for tempering and
				                    converting to float. */
    unsigned int mask;		/**< This is a mask for state space */
    unsigned char poly_sha1[21]; /**< SHA1 digest */
        
    FQUALIFIERS
    ~mtgp32_fast_param() { }
};
    
struct mtgp32_state
{
    unsigned int status[MTGP_LS];
    int offset;
    int id;
       
    FQUALIFIERS
    ~mtgp32_state() { }
};
    
void rocrand_mtgp32_init_state(unsigned int array[],
		               const mtgp32_fast_param *para, unsigned int seed) 
{
    int i;
    int size = para->mexp / 32 + 1;
    unsigned int hidden_seed;
    unsigned int tmp;
    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(array, tmp & 0xff, sizeof(unsigned int) * size);
    array[0] = seed;
    array[1] = hidden_seed;
    for (i = 1; i < size; i++) 
	    array[i] ^= (1812433253) * (array[i - 1] ^ (array[i - 1] >> 30)) + i;
    
}
    
rocrand_status rocrand_make_state_mtgp32(mtgp32_state * d_state,
				                 mtgp32_fast_param params[],
				                 int n,
                                 unsigned long long seed)
{
    int i;
    mtgp32_state* h_state = (mtgp32_state *) malloc(sizeof(mtgp32_state) * n);
    seed = seed ^ (seed >> 32);

    if (h_state == NULL) 
	    return ROCRAND_STATUS_LAUNCH_FAILURE;
    
    for (i = 0; i < n; i++) {
	    rocrand_mtgp32_init_state(&(h_state[i].status[0]), &params[i], (unsigned int)seed + i + 1);
        h_state[i].offset = 0;
        h_state[i].id = i;
    }
    
    hipMemcpy(d_state, h_state, sizeof(mtgp32_state) * n,
		      hipMemcpyHostToDevice);
    free(h_state);
    
    if(hipPeekAtLastError() != hipSuccess)
        return ROCRAND_STATUS_LAUNCH_FAILURE;
    
    return ROCRAND_STATUS_SUCCESS;
}
    
rocrand_status rocrand_make_constant(const mtgp32_fast_param params[], mtgp32_param * p) 
{
    const int block_num = MTGP_BN_MAX;
    const int size1 = sizeof(unsigned int) * block_num;
    const int size2 = sizeof(unsigned int) * block_num * MTGP_TS;
    unsigned int *h_pos_tbl;
    unsigned int *h_sh1_tbl;
    unsigned int *h_sh2_tbl;
    unsigned int *h_param_tbl;
    unsigned int *h_temper_tbl;
    unsigned int *h_single_temper_tbl;
    unsigned int *h_mask;
    rocrand_status status = ROCRAND_STATUS_SUCCESS;
    
    h_pos_tbl = (unsigned int *)malloc(size1);
    h_sh1_tbl = (unsigned int *)malloc(size1);
    h_sh2_tbl = (unsigned int *)malloc(size1);
    h_param_tbl = (unsigned int *)malloc(size2);
    h_temper_tbl = (unsigned int *)malloc(size2);
    h_single_temper_tbl = (unsigned int *)malloc(size2);
    h_mask = (unsigned int *)malloc(sizeof(unsigned int));
    if (h_pos_tbl == NULL
	    || h_sh1_tbl == NULL
	    || h_sh2_tbl == NULL
	    || h_param_tbl == NULL
	    || h_temper_tbl == NULL
	    || h_single_temper_tbl == NULL
	    || h_mask == NULL) {
        if (h_pos_tbl != NULL) free(h_pos_tbl);
        if (h_sh1_tbl != NULL) free(h_sh1_tbl);
        if (h_sh2_tbl != NULL) free(h_sh2_tbl);
        if (h_param_tbl != NULL) free(h_param_tbl);
        if (h_temper_tbl != NULL) free(h_temper_tbl);
        if (h_single_temper_tbl != NULL) free(h_single_temper_tbl);
        if (h_mask != NULL) free(h_mask);
        status = ROCRAND_STATUS_ALLOCATION_FAILED;
    } else {       

        h_mask[0] = params[0].mask;
        for (int i = 0; i < block_num; i++) {
	        h_pos_tbl[i] = params[i].pos;
	        h_sh1_tbl[i] = params[i].sh1;
	        h_sh2_tbl[i] = params[i].sh2;
	        for (int j = 0; j < MTGP_TS; j++) {
	            h_param_tbl[i * MTGP_TS + j] = params[i].tbl[j];
	            h_temper_tbl[i * MTGP_TS + j] = params[i].tmp_tbl[j];
	            h_single_temper_tbl[i * MTGP_TS + j] = params[i].flt_tmp_tbl[j];
	        }
        }
        if (hipMemcpy( p->pos_tbl, 
                        h_pos_tbl, size1, hipMemcpyHostToDevice) != hipSuccess)
        { 
            status = ROCRAND_STATUS_ALLOCATION_FAILED;
        } else
        if (hipMemcpy( p->sh1_tbl, 
                        h_sh1_tbl, size1, hipMemcpyHostToDevice) != hipSuccess)
        {
            status = ROCRAND_STATUS_ALLOCATION_FAILED;
        } else
        if (hipMemcpy( p->sh2_tbl, 
                        h_sh2_tbl, size1, hipMemcpyHostToDevice) != hipSuccess)
        {
            status = ROCRAND_STATUS_ALLOCATION_FAILED;
        } else
        if (hipMemcpy( p->param_tbl, 
                        h_param_tbl, size2, hipMemcpyHostToDevice) != hipSuccess)
        {
            status = ROCRAND_STATUS_ALLOCATION_FAILED;
        } else
        if (hipMemcpy( p->temper_tbl, 
                        h_temper_tbl, size2, hipMemcpyHostToDevice) != hipSuccess)
        {
            status = ROCRAND_STATUS_ALLOCATION_FAILED;
        } else
        if (hipMemcpy( p->single_temper_tbl, 
                        h_single_temper_tbl, size2, hipMemcpyHostToDevice) != hipSuccess)
        {
            status = ROCRAND_STATUS_ALLOCATION_FAILED;
        } else
        if (hipMemcpy( p->mask, 
                        h_mask, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        {
            status = ROCRAND_STATUS_ALLOCATION_FAILED;
        } 
    }
    if (h_pos_tbl != NULL) free(h_pos_tbl);
    if (h_sh1_tbl != NULL) free(h_sh1_tbl);
    if (h_sh2_tbl != NULL) free(h_sh2_tbl);
    if (h_param_tbl != NULL) free(h_param_tbl);
    if (h_temper_tbl != NULL) free(h_temper_tbl);
    if (h_single_temper_tbl != NULL)free(h_single_temper_tbl);
    if (h_mask != NULL) free(h_mask);
    return status;
}

class mtgp32_engine
{
public:
    FQUALIFIERS
    mtgp32_engine()
    {

    }

    FQUALIFIERS
    mtgp32_engine(const mtgp32_state m_state,
                  const mtgp32_param m_param)
    {
        this->m_state = m_state;
        this->m_param = m_param;
    }

    FQUALIFIERS
    ~mtgp32_engine() { }

    FQUALIFIERS
    unsigned int operator()()
    {
        return this->next();
    }

    FQUALIFIERS
    unsigned int next()
    {
        #if defined(__HIP_DEVICE_COMPILE__)
        unsigned int t = (hipBlockIdx_z * hipBlockIdx_y * hipThreadIdx_z) + 
                         (hipBlockIdx_x * hipThreadIdx_y) + hipThreadIdx_x;
        unsigned int d = hipBlockIdx_z * hipBlockIdx_y * hipBlockIdx_x;
        int pos = m_param.pos_tbl[m_state.id];
        unsigned int r;
        unsigned int o;
        
        r = para_rec(m_state.status[(t + m_state.offset) & MTGP_LS],
                     m_state.status[(t + m_state.offset + 1) & MTGP_LS],
                     m_state.status[(t + m_state.offset + pos) & MTGP_LS],
                     m_state.id);
        m_state.status[(t + m_state.offset + MTGP_N) & MTGP_LS] = r;
        
        o = temper(r, m_state.status[(t + m_state.offset + pos -1) & MTGP_LS], m_state.id);
        __syncthreads();
        if (t == 0)
            m_state.offset = (m_state.offset + d) & MTGP_LS;
        __syncthreads();
        return o;
        #endif
        return 0;
    }

private:
    FQUALIFIERS 
    unsigned int para_rec(unsigned int X1, unsigned int X2, unsigned int Y, int bid) 
    {
        unsigned int X = (X1 & m_param.mask[0]) ^ X2;
        unsigned int MAT;

        X ^= X << m_param.sh1_tbl[bid];
        Y = X ^ (Y >> m_param.sh2_tbl[bid]);
        MAT = m_param.param_tbl[bid][Y & 0x0f];
        return Y ^ MAT;
    }
    
    FQUALIFIERS
    unsigned int temper(unsigned int V, unsigned int T, int bid) 
    {
        unsigned int MAT;

        T ^= T >> 16;
        T ^= T >> 8;
        MAT = m_param.temper_tbl[bid][T & 0x0f];
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
    return state->next();
}

#endif // ROCRAND_MTGP32_H_
