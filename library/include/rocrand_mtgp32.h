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
 * Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University.  All rights reserved.
 * Copyright (c) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
 * University and University of Tokyo.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of the Hiroshima University nor the names of
 *       its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written
 *       permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
#define MTGP_STATE 1024
#define MTGP_MASK 1023


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
    int mexp;			
    int pos;			
    int sh1;			
    int sh2;			
    unsigned int tbl[16];		
    unsigned int tmp_tbl[16];	
    unsigned int flt_tmp_tbl[16];	
    unsigned int mask;		
    unsigned char poly_sha1[21]; 
        
    FQUALIFIERS
    ~mtgp32_fast_param() { }
};
    
struct mtgp32_state
{
    unsigned int status[MTGP_STATE];
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
    
    hipMemcpy(d_state, h_state, sizeof(mtgp32_state) * n, hipMemcpyHostToDevice);
    free(h_state);
    
    if (hipPeekAtLastError() != hipSuccess)
        return ROCRAND_STATUS_LAUNCH_FAILURE;
    
    return ROCRAND_STATUS_SUCCESS;
}
    
rocrand_status rocrand_make_constant(const mtgp32_fast_param params[], mtgp32_param * p) 
{
    const int block_num = MTGP_BN_MAX;
    const int size1 = sizeof(uint32_t) * block_num;
    const int size2 = sizeof(uint32_t) * block_num * MTGP_TS;
    uint32_t *h_pos_tbl;
    uint32_t *h_sh1_tbl;
    uint32_t *h_sh2_tbl;
    uint32_t *h_param_tbl;
    uint32_t *h_temper_tbl;
    uint32_t *h_single_temper_tbl;
    uint32_t *h_mask;
    h_pos_tbl = (uint32_t *)malloc(size1);
    h_sh1_tbl = (uint32_t *)malloc(size1);
    h_sh2_tbl = (uint32_t *)malloc(size1);
    h_param_tbl = (uint32_t *)malloc(size2);
    h_temper_tbl = (uint32_t *)malloc(size2);
    h_single_temper_tbl = (uint32_t *)malloc(size2);
    h_mask = (uint32_t *)malloc(sizeof(uint32_t));
    rocrand_status status = ROCRAND_STATUS_SUCCESS;
    
    if (h_pos_tbl == NULL || h_sh1_tbl == NULL || h_sh2_tbl == NULL
	    || h_param_tbl == NULL || h_temper_tbl == NULL || h_single_temper_tbl == NULL
	    || h_mask == NULL) {
	    printf("failure in allocating host memory for constant table.\n");
	    return ROCRAND_STATUS_ALLOCATION_FAILED;
    }
    
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
    
    if (hipMemcpy(p->pos_tbl, h_pos_tbl, size1, hipMemcpyHostToDevice) != hipSuccess)
        status = ROCRAND_STATUS_ALLOCATION_FAILED;
    if (hipMemcpy(p->sh1_tbl, h_sh1_tbl, size1, hipMemcpyHostToDevice) != hipSuccess)
        status = ROCRAND_STATUS_ALLOCATION_FAILED;
    if (hipMemcpy(p->sh2_tbl, h_sh2_tbl, size1, hipMemcpyHostToDevice) != hipSuccess)
        status = ROCRAND_STATUS_ALLOCATION_FAILED;
    if (hipMemcpy(p->param_tbl, h_param_tbl, size2, hipMemcpyHostToDevice) != hipSuccess)
        status = ROCRAND_STATUS_ALLOCATION_FAILED;
    if (hipMemcpy(p->temper_tbl, h_temper_tbl, size2, hipMemcpyHostToDevice) != hipSuccess)
        status = ROCRAND_STATUS_ALLOCATION_FAILED;
    if (hipMemcpy(p->single_temper_tbl, h_single_temper_tbl, size2, hipMemcpyHostToDevice) != hipSuccess)
        status = ROCRAND_STATUS_ALLOCATION_FAILED;
    if (hipMemcpy(p->mask, h_mask, sizeof(unsigned int), hipMemcpyHostToDevice) != hipSuccess)
        status = ROCRAND_STATUS_ALLOCATION_FAILED;
    
    free(h_pos_tbl);
    free(h_sh1_tbl);
    free(h_sh2_tbl);
    free(h_param_tbl);
    free(h_temper_tbl);
    free(h_single_temper_tbl);
    free(h_mask);
    
    return status;
}

class mtgp32_engine
{
public:
    struct mtgp32_kernel_param
    {
        unsigned int pos_tbl;
        unsigned int param_tbl[MTGP_TS];
        unsigned int temper_tbl[MTGP_TS];
        unsigned int single_temper_tbl[MTGP_TS];
        unsigned int sh1_tbl;
        unsigned int sh2_tbl;
        unsigned int mask;

        FQUALIFIERS
        ~mtgp32_kernel_param() { }
    };
    
    FQUALIFIERS
    mtgp32_engine()
    {

    }

    FQUALIFIERS
    mtgp32_engine(const mtgp32_state m_state,
                  const mtgp32_param * params,
                  int bid)
    {
        this->m_state = m_state;
        m_param.pos_tbl = params->pos_tbl[bid];
        m_param.sh1_tbl = params->sh1_tbl[bid];
        m_param.sh2_tbl = params->sh2_tbl[bid];
        m_param.mask = params->mask[0];
	    for (int j = 0; j < MTGP_TS; j++) {
	        m_param.param_tbl[j] = params->param_tbl[bid][j];
	        m_param.temper_tbl[j] = params->temper_tbl[bid][j];
	        m_param.single_temper_tbl[j] = params->single_temper_tbl[bid][j];
	    }
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
        unsigned int t = (hipBlockDim_z * hipBlockDim_y * hipThreadIdx_z) + 
                         (hipBlockDim_x * hipThreadIdx_y) + hipThreadIdx_x;
        unsigned int d = hipBlockDim_z * hipBlockDim_y * hipBlockDim_x;
        int pos = m_param.pos_tbl;
        unsigned int r;
        unsigned int o;
        
        r = para_rec(m_state.status[(t + m_state.offset) & MTGP_MASK],
                     m_state.status[(t + m_state.offset + 1) & MTGP_MASK],
                     m_state.status[(t + m_state.offset + pos) & MTGP_MASK]);
        m_state.status[(t + m_state.offset + MTGP_N) & MTGP_MASK] = r;
        
        o = temper(r, m_state.status[(t + m_state.offset + pos - 1) & MTGP_MASK]);
        __syncthreads();
        if (t == 0)
            m_state.offset = (m_state.offset + d) & MTGP_MASK;
        __syncthreads();
        return o;
        #else
        return 0;
        #endif
    }
    
    FQUALIFIERS
    unsigned int next_single()
    {
        #if defined(__HIP_DEVICE_COMPILE__)
        unsigned int t = (hipBlockDim_z * hipBlockDim_y * hipThreadIdx_z) + 
                         (hipBlockDim_x * hipThreadIdx_y) + hipThreadIdx_x;
        unsigned int d = hipBlockDim_z * hipBlockDim_y * hipBlockDim_x;
        int pos = m_param.pos_tbl;
        unsigned int r;
        unsigned int o;
        
        r = para_rec(m_state.status[(t + m_state.offset) & MTGP_MASK],
                     m_state.status[(t + m_state.offset + 1) & MTGP_MASK],
                     m_state.status[(t + m_state.offset + pos) & MTGP_MASK]);
        m_state.status[(t + m_state.offset + MTGP_N) & MTGP_MASK] = r;
        
        o = temper_single(r, m_state.status[(t + m_state.offset + pos - 1) & MTGP_MASK]);
        __syncthreads();
        if (t == 0)
            m_state.offset = (m_state.offset + d) & MTGP_MASK;
        __syncthreads();
        return o;
        #else
        return 0;
        #endif
    }

private:
    FQUALIFIERS 
    unsigned int para_rec(unsigned int X1, unsigned int X2, unsigned int Y) 
    {
        unsigned int X = (X1 & m_param.mask) ^ X2;
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
    
    FQUALIFIERS
    unsigned int temper_single(unsigned int V, unsigned int T) 
    {
        unsigned int MAT;
        unsigned int r;

        T ^= T >> 16;
        T ^= T >> 8;
        MAT = m_param.single_temper_tbl[T & 0x0f];
        r = (V >> 9) ^ MAT;
        return r;
    }
    
protected:
    // State
    mtgp32_state m_state;
    mtgp32_kernel_param m_param;

}; // mtgp32_engine class

} // end namespace rocrand_device

typedef rocrand_device::mtgp32_engine rocrand_state_mtgp32;

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
