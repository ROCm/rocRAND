// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

// Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//  3. The names of its contributors may not be used to endorse or promote
//     products derived from this software without specific prior written
//     permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROCRAND_RNG_MT19937_H_
#define ROCRAND_RNG_MT19937_H_

#define MT_FQUALIFIERS __forceinline__ __device__

#include <hip/hip_runtime.h>

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_mt19937_precomputed.h>

#include "common.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"

namespace
{
/// Number of independent generators. Value is fixed to produce deterministic number stream.
static constexpr unsigned int generator_count = 8192;
/// Number of threads that cooperate to run one generator. Value is fixed in implementation.
static constexpr unsigned int threads_per_generator = 8;
/// Number of threads per block. Can be tweaked for performance.
static constexpr unsigned int thread_count = 256;
/// Number of threads per block for jump-ahead kernel. Can be tweaked for performance.
static constexpr unsigned int jump_ahead_thread_count = 128;

static_assert(thread_count % threads_per_generator == 0,
              "All eight threads of the generator must be in the same block");
static_assert(generator_count == (1 << mt19937_jump_powers),
              "Not enough mt19937_jump values to initialize all generators");
} // namespace

namespace rocrand_host
{
namespace detail
{

namespace
{
// MT19937 constants.

/// Number of elements in the state vector.
static constexpr unsigned int n = 624;
/// Exponent of Mersenne prime.
static constexpr unsigned int mexp = 19937;
/// The next value of element \p i depends on <tt>(i + 1) % n</tt> and <tt>(i + m) % n</tt>.
static constexpr unsigned int m = 397;
/// Vector constant used in masking operation to represent multiplication by matrix A.
static constexpr unsigned int matrix_a = 0x9908B0DFU;
/// Mask to select the most significant <tt>w - r</tt> bits from a \p w bits word, where
/// \p w is the number of bits per generated word (32) and \p r is an algorithm constant.
static constexpr unsigned int upper_mask = 0x80000000U;
/// Mask to select the least significant \p r bits from a \p w bits word, where
/// \p w is the number of bits per generated word (32) and \p r is an algorithm constant.
static constexpr unsigned int lower_mask = 0x7FFFFFFFU;
} // namespace

struct mt19937_octo_engine
{
    /// Tuples of \p items_per_thread follow a regular pattern.
    static constexpr unsigned int items_per_thread = 7U;

    struct mt19937_octo_state
    {
        /// Thread 0 has element   0, thread 1 has element 113, thread 2 has element 170,
        /// thread 3 had element 283, thread 4 has element 340, thread 5 has element 397,
        /// thread 6 has element 510, thread 7 has element 567.
        /// Thread i for i in [0, 7) has the following elements (ipt = items_per_thread):
        /// [  1 + ipt * i,   1 + ipt * (i + 1)), [398 + ipt * i, 398 + ipt * (i + 1)), [171 + ipt * i, 171 + ipt * (i + 1)),
        /// [568 + ipt * i, 568 + ipt * (i + 1)), [341 + ipt * i, 341 + ipt * (i + 1)), [114 + ipt * i, 114 + ipt * (i + 1)),
        /// [511 + ipt * i, 511 + ipt * (i + 1)), [284 + ipt * i, 284 + ipt * (i + 1)), [ 57 + ipt * i,  57 + ipt * (i + 1)),
        /// [454 + ipt * i, 454 + ipt * (i + 1)), [227 + ipt * i, 227 + ipt * (i + 1))
        ///
        /// which are 1 + 11 * 7 = 78 values per thread.
        unsigned int mt[1U + items_per_thread * 11U];
        /// The index of the next value to be returned in global array of values.
        /// The actual returned values are of a different order per n elements.
        /// When \p mti is <tt>n</tt>, \p n new values must be calculated.
        unsigned int mti;
    };

    /// Constants to map the indices to \p mt19937_octo_state.mt_extra indices.
    /// For example, \p i000_0 is the index of \p 0 owned by thread 0.

    static constexpr unsigned int i000_0 = 0;
    static constexpr unsigned int i113_1 = 0;
    static constexpr unsigned int i170_2 = 0;
    static constexpr unsigned int i283_3 = 0;
    static constexpr unsigned int i340_4 = 0;
    static constexpr unsigned int i397_5 = 0;
    static constexpr unsigned int i510_6 = 0;
    static constexpr unsigned int i567_7 = 0;

    /// Constants used to map the indices to \p mt19937_octo_state.mt indices.
    /// For example, \p i001 is the index of <tt>1 + tid * ipt</tt>.

    static constexpr unsigned int i001 = 1 + items_per_thread * 0;
    static constexpr unsigned int i057 = 1 + items_per_thread * 1;
    static constexpr unsigned int i114 = 1 + items_per_thread * 2;
    static constexpr unsigned int i171 = 1 + items_per_thread * 3;
    static constexpr unsigned int i227 = 1 + items_per_thread * 4;
    static constexpr unsigned int i284 = 1 + items_per_thread * 5;
    static constexpr unsigned int i341 = 1 + items_per_thread * 6;
    static constexpr unsigned int i398 = 1 + items_per_thread * 7;
    static constexpr unsigned int i454 = 1 + items_per_thread * 8;
    static constexpr unsigned int i511 = 1 + items_per_thread * 9;
    static constexpr unsigned int i568 = 1 + items_per_thread * 10;

    /// Initialize the octo engine from the engine it shares with seven other threads.
    MT_FQUALIFIERS void gather(const unsigned int engine[n])
    {
        constexpr unsigned int off_cnt = 11;
        /// Used to map the \p mt19937_octo_state.mt indices to \p mt19937_state.mt indices.
        static constexpr unsigned int offsets[off_cnt]
            = {1, 57, 114, 171, 227, 284, 341, 398, 454, 511, 568};

        const unsigned int tid = threadIdx.x & 7U;

        // initialize the elements that follow a regular pattern
        for(unsigned int i = 0; i < off_cnt; i++)
        {
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                const unsigned int index = offsets[i] + items_per_thread * tid + j;
                // + 1 for the special value
                m_state.mt[1U + items_per_thread * i + j] = engine[index];
            }
        }

        // initialize the elements that do not follow a regular pattern
        constexpr unsigned int dest_idx[threads_per_generator]
            = {i000_0, i113_1, i170_2, i283_3, i340_4, i397_5, i510_6, i567_7};
        constexpr unsigned int src_idx[threads_per_generator]
            = {0, 113, 170, 283, 340, 397, 510, 567};
        m_state.mt[dest_idx[tid]] = engine[src_idx[tid]];

        // set to n, to indicate that a batch of n values can be calculated at a time
        m_state.mti = n;
    }

    /// Returns \p val from thread <tt>tid mod 8</tt>.
    static MT_FQUALIFIERS unsigned int shuffle(unsigned int val, unsigned int tid)
    {
        return __shfl(val, tid, 8);
    }

    /// For thread i, returns \p val from thread <tt>(i + 1) mod 8</tt>
    static MT_FQUALIFIERS unsigned int shuffle_down(unsigned int val)
    {
        return __shfl_down(val, 1, 8);
    }

    /// For thread i, returns \p val from thread <tt>(i - 1) mod 8</tt>
    static MT_FQUALIFIERS unsigned int shuffle_up(unsigned int val)
    {
        return __shfl_up(val, 1, 8);
    }
    /// Calculates value of index \p i using values <tt>i</tt>, <tt>(i + 1) % n</tt>, and <tt>(i + m) % n</tt>.
    static MT_FQUALIFIERS unsigned int
        comp(unsigned int mt_i, unsigned int mt_i_1, unsigned int mt_i_m)
    {
        const unsigned int y   = (mt_i & upper_mask) | (mt_i_1 & lower_mask);
        const unsigned int mag = (y & 0x1U) * matrix_a;
        return mt_i_m ^ (y >> 1) ^ mag;
    }

    /// Thread \p tid computes      [i + ipt * tid,     i + ipt * (tid + 1)).
    /// This computation depends on [i + ipt * tid + 1, i + ipt * (tid + 1) + 1) and
    ///                             [i + ipt * tid + m, i + ipt * (tid + 1) + m).
    /// \p idx_i is the local address of <tt>i</tt>: <tt>i + ipt * tid</tt>.
    /// \p idx_m is the local address of <tt>m</tt>: <tt>i + ipt * tid + m</tt>.
    /// \p last_dep_tid_7 is the value of <tt>i + ipt * (tid + 1)</tt>, which is
    /// required as it is the only value not owned by thread <tt>pid</tt>.
    MT_FQUALIFIERS void comp_vector(unsigned int tid,
                                    unsigned int idx_i,
                                    unsigned int idx_m,
                                    unsigned int last_dep_tid_7)
    {
        // communicate the dependency for the last value
        unsigned int last_dep = shuffle_down(m_state.mt[idx_i]);
        // thread 7 needs a special value that does not fit the pattern
        last_dep = tid == 7 ? last_dep_tid_7 : last_dep;

        unsigned int j;
        for(j = 0; j < items_per_thread - 1; j++)
        {
            // compute (i + ipt * tid + j)': needs (i + ipt * tid + 1 + j) % n and (i + ipt * tid + m + j) % n
            m_state.mt[idx_i + j]
                = comp(m_state.mt[idx_i + j], m_state.mt[idx_i + j + 1], m_state.mt[idx_m + j]);
        }
        // compute the last value using the communicated dependency
        m_state.mt[idx_i + j] = comp(m_state.mt[idx_i + j], last_dep, m_state.mt[idx_m + j]);
    }

    /// Eights threads collaborate in computing the n next values.
    MT_FQUALIFIERS void gen_next_n()
    {
        const unsigned int tid = threadIdx.x & 7U;

        // compute eleven vectors that follow a regular pattern and compute
        // eight special values for a total of n new elements.
        // ' indicates new value

        // compute   0': needs   1 and 397
        const unsigned int v397 = shuffle(m_state.mt[i397_5], 5);
        if(tid == 0)
        {
            m_state.mt[i000_0] = comp(m_state.mt[i000_0], m_state.mt[i001], v397);
        }

        // compute [  1 + i * ipt,   1 + ipt * (i + 1))' = [  1,  56]':
        // needs [  1,  57] and [398, 453]
        const unsigned int v057 = shuffle(m_state.mt[i057], 0);
        comp_vector(tid, i001, i398, v057);

        // compute [ 57 + i * ipt,  57 + ipt * (i + 1))' = [ 57, 112]':
        // needs [ 57, 113] and [454, 509]
        const unsigned int v113 = shuffle(m_state.mt[i113_1], 1);
        comp_vector(tid, i057, i454, v113);

        // compute 113': needs 114 and 510
        const unsigned int v114 = shuffle(m_state.mt[i114], 0);
        const unsigned int v510 = shuffle(m_state.mt[i510_6], 6);
        if(tid == 1)
        {
            m_state.mt[i113_1] = comp(m_state.mt[i113_1], v114, v510);
        }

        // compute [114 + i * ipt, 114 + ipt * (i + 1))' = [114, 169]':
        // needs [114, 170] and [511, 566]
        const unsigned int v170 = shuffle(m_state.mt[i170_2], 2);
        comp_vector(tid, i114, i511, v170);

        // compute 170': needs 171 and 567
        const unsigned int v171 = shuffle(m_state.mt[i171], 0);
        const unsigned int v567 = shuffle(m_state.mt[i567_7], 7);
        if(tid == 2)
        {
            m_state.mt[i170_2] = comp(m_state.mt[i170_2], v171, v567);
        }

        // compute [171 + i * ipt, 171 + ipt * (i + 1))' = [171, 226]':
        // needs [171, 227] and [568, 623]
        const unsigned int v227 = shuffle(m_state.mt[i227], 0);
        comp_vector(tid, i171, i568, v227);

        // compute [227 + i * ipt, 227 + ipt * (i + 1))' = [227, 282]':
        // needs [227, 283] and [  0,  55]'
        const unsigned int v283 = shuffle(m_state.mt[i283_3], 3);
        // comp_vector(tid, i227, s_000, v283);
        // written out below, since value 0 does not fit the regular pattern
        {
            unsigned int last_dep = shuffle_down(m_state.mt[i227]);
            last_dep              = tid == 7 ? v283 : last_dep;

            // communicate the dependency for the first value
            unsigned int first_dep = shuffle_up(m_state.mt[i001 + items_per_thread - 1]);
            first_dep              = tid == 0 ? m_state.mt[i000_0] : first_dep;

            // extract the first and last iterations from the loop
            unsigned int j       = 0;
            m_state.mt[i227 + j] = comp(m_state.mt[i227 + j], m_state.mt[i227 + j + 1], first_dep);
            for(j = 1; j < items_per_thread - 1; j++)
            {
                m_state.mt[i227 + j] = comp(m_state.mt[i227 + j],
                                            m_state.mt[i227 + j + 1],
                                            m_state.mt[i001 + j - 1]);
            }
            m_state.mt[i227 + j] = comp(m_state.mt[i227 + j], last_dep, m_state.mt[i001 + j - 1]);
        }

        // compute 283': needs 284 and  56'
        const unsigned int v284 = shuffle(m_state.mt[i284], 0);
        const unsigned int v056 = shuffle(m_state.mt[i001 + 6], 7); // 1 + 7 * 7 + 6 = 56
        if(tid == 3)
        {
            m_state.mt[i283_3] = comp(m_state.mt[i283_3], v284, v056);
        }

        // compute [284 + i * ipt, 284 + ipt * (i + 1))' = [284, 339]':
        // needs [284, 340] and [ 57, 112]'
        const unsigned int v340 = shuffle(m_state.mt[i340_4], 4);
        comp_vector(tid, i284, i057, v340);

        // compute 340': needs 341 and 113'
        const unsigned int v113_ = shuffle(m_state.mt[i113_1], 1);
        const unsigned int v341  = shuffle(m_state.mt[i341], 0);
        if(tid == 4)
        {
            m_state.mt[i340_4] = comp(m_state.mt[i340_4], v341, v113_);
        }

        // compute [341 + i * ipt, 341 + ipt * (i + 1))' = [341, 396]':
        // needs [341, 397] and [114, 169]'
        const unsigned int v397_ = shuffle(m_state.mt[i397_5], 5);
        comp_vector(tid, i341, i114, v397_);

        // compute 397': needs 398 and 170'
        const unsigned int v398  = shuffle(m_state.mt[i398], 0);
        const unsigned int v170_ = shuffle(m_state.mt[i170_2], 2);
        if(tid == 5)
        {
            m_state.mt[i397_5] = comp(m_state.mt[i397_5], v398, v170_);
        }

        // compute [398 + i * ipt, 398 + ipt * (i + 1))' = [398, 453]':
        // needs [398, 454] and [171, 226]'
        const unsigned int v454 = shuffle(m_state.mt[i454], 0);
        comp_vector(tid, i398, i171, v454);

        // compute [454 + i * ipt, 454 + ipt * (i + 1))' = [454, 509]':
        // needs [454, 510] and [227, 282]'
        const unsigned int v510_ = shuffle(m_state.mt[i510_6], 6);
        comp_vector(tid, i454, i227, v510_);

        // compute 510': needs 511 and 283'
        const unsigned int v511  = shuffle(m_state.mt[i511], 0);
        const unsigned int v283_ = shuffle(m_state.mt[i283_3], 3);
        if(tid == 6)
        {
            m_state.mt[i510_6] = comp(m_state.mt[i510_6], v511, v283_);
        }

        // compute [511 + i * ipt, 511 + ipt * (i + 1))' = [511, 566]':
        // needs [511, 567] and [284, 339]'
        const unsigned int v567_ = shuffle(m_state.mt[i567_7], 7);
        comp_vector(tid, i511, i284, v567_);

        // compute 567': needs 568 and 340'
        const unsigned int v568 = shuffle(m_state.mt[i568], 0);
        const unsigned int i340 = shuffle(m_state.mt[i340_4], 4);
        if(tid == 7)
        {
            m_state.mt[i567_7] = comp(m_state.mt[i567_7], v568, i340);
        }

        // compute [568 + i * ipt, 568 + ipt * (i + 1))' = [568, 623]':
        // needs [568, 623], [0, 0]', and [341, 396]'
        const unsigned int v000 = shuffle(m_state.mt[i000_0], 0);
        comp_vector(tid, i568, i341, v000);

        m_state.mti = 0;
    }

    /// Every thread produces one value, must be called for all eight threads
    /// at the same time.
    MT_FQUALIFIERS unsigned int operator()()
    {
        if(m_state.mti == n)
        {
            gen_next_n();
        }

        unsigned int y = m_state.mt[m_state.mti / threads_per_generator];
        m_state.mti += threads_per_generator;

        // perform tempering on y

        constexpr unsigned int TEMPERING_MASK_B = 0x9D2C5680U;
        constexpr unsigned int TEMPERING_MASK_C = 0xEFC60000U;

        y ^= y >> 11; /* tempering shift u */
        y ^= (y << 7) /* tempering shift s */ & TEMPERING_MASK_B;
        y ^= (y << 15) /* tempering shift t */ & TEMPERING_MASK_C;
        y ^= y >> 18; /* tempering shift l */

        return y;
    }

private:
    mt19937_octo_state m_state;
};

/// Computes i % n, i must be in range [0, 2 * n)
MT_FQUALIFIERS unsigned int wrap_n(unsigned int i)
{
    return i - (i < n ? 0 : n);
}

ROCRAND_KERNEL
__launch_bounds__(jump_ahead_thread_count) void jump_ahead_kernel(
    unsigned int* __restrict__ engines,
    unsigned long long seed,
    const unsigned int* __restrict__ jump)
{
    constexpr unsigned int block_size       = jump_ahead_thread_count;
    constexpr unsigned int items_per_thread = (n + block_size - 1) / block_size;
    constexpr unsigned int tail_n           = n - (items_per_thread - 1) * block_size;

    __shared__ unsigned int temp[n];
    unsigned int            state[items_per_thread];

    // Initialize state 0 (engine_id = 0) used as a base for all engines.
    // It uses a recurrence relation so one thread calculates all n values.
    if(threadIdx.x == 0)
    {
        const unsigned int seedu = (seed >> 32) ^ seed;
        temp[0]                  = seedu;
        for(unsigned int i = 1; i < n; i++)
        {
            temp[i] = 1812433253 * (temp[i - 1] ^ (temp[i - 1] >> 30)) + i;
        }
    }
    __syncthreads();

    for(unsigned int i = 0; i < items_per_thread; i++)
    {
        if(i < items_per_thread - 1 || threadIdx.x < tail_n) // Check only for the last iteration
        {
            state[i] = temp[i * block_size + threadIdx.x];
        }
    }
    __syncthreads();

    const unsigned int engine_id = blockIdx.x;

    // Jump ahead by engine_id * 2 ^ 1000 using precomputed polynomials for 2 ^ p * 2 ^ 1000
    for(unsigned int p = 0; p < mt19937_jump_powers; p++)
    {
        if((engine_id & (1 << p)) == 0)
        {
            continue;
        }

        // Compute jumping ahead with standard Horner method

        unsigned int ptr = 0;
        for(unsigned int i = threadIdx.x; i < n; i += block_size)
        {
            temp[i] = 0;
        }
        __syncthreads();

        const unsigned int* pf = jump + p * mt19937_p_size;
        for(int pfi = mexp - 1; pfi >= 0; pfi--)
        {
            // Generate next state
            if(threadIdx.x == 0)
            {
                unsigned int t0 = temp[ptr];
                unsigned int t1 = temp[wrap_n(ptr + 1)];
                unsigned int tm = temp[wrap_n(ptr + m)];
                unsigned int y  = (t0 & upper_mask) | (t1 & lower_mask);
                temp[ptr]       = tm ^ (y >> 1) ^ ((y & 0x1U) ? matrix_a : 0);
            }
            __syncthreads();
            ptr = wrap_n(ptr + 1);

            if((pf[pfi / 32] >> (pfi % 32)) & 1)
            {
                // Add state to temp
                for(unsigned int i = 0; i < items_per_thread; i++)
                {
                    if(i < items_per_thread - 1 || threadIdx.x < tail_n)
                    {
                        temp[wrap_n(ptr + i * block_size + threadIdx.x)] ^= state[i];
                    }
                }
                __syncthreads();
            }
        }

        // Jump of the next power of 2 will be applied to the current state
        for(unsigned int i = 0; i < items_per_thread; i++)
        {
            if(i < items_per_thread - 1 || threadIdx.x < tail_n)
            {
                state[i] = temp[wrap_n(ptr + i * block_size + threadIdx.x)];
            }
        }
        __syncthreads();
    }

    // Save state
    for(unsigned int i = 0; i < items_per_thread; i++)
    {
        if(i < items_per_thread - 1 || threadIdx.x < tail_n)
        {
            engines[engine_id * n + i * block_size + threadIdx.x] = state[i];
        }
    }
}

ROCRAND_KERNEL
__launch_bounds__(thread_count) void init_engines_kernel(
    mt19937_octo_engine* __restrict__ octo_engines, const unsigned int* __restrict__ engines)
{
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // every eight octo engines gather from the same engine
    octo_engines[thread_id].gather(&engines[thread_id / threads_per_generator * n]);
}

template<class T, class Distribution>
ROCRAND_KERNEL
    __launch_bounds__(thread_count) void generate_kernel(mt19937_octo_engine* __restrict__ engines,
                                                         T* __restrict__ data,
                                                         const size_t size,
                                                         Distribution distribution)
{
    constexpr unsigned int input_width  = Distribution::input_width;
    constexpr unsigned int output_width = Distribution::output_width;

    // every eight threads together produce eight values
    constexpr unsigned int full_output_width = threads_per_generator * output_width;

    using vec_type = aligned_vec_type<T, output_width>;

    const unsigned int     thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr unsigned int stride    = threads_per_generator * generator_count;

    unsigned int input[input_width];
    T            output[output_width];

    // find a vec_type boundary that is a multiple of threads_per_generator

    const uintptr_t uintptr = reinterpret_cast<uintptr_t>(data);
    // uintptr + misalignment = nearest boundary in elements T
    const size_t misalignment
        = (full_output_width - (uintptr / sizeof(T)) % full_output_width) % full_output_width;
    // number of elements T before the boundary
    const unsigned int head_size = min(size, misalignment);
    // number of elements T after the boundary + full elements
    const unsigned int tail_size = (size - head_size) % full_output_width;
    // number of boundary-aligned elements vec_type that is a multiple of thread_per_generator
    const size_t vec_n = (size - head_size) / full_output_width * threads_per_generator;

    mt19937_octo_engine engine = engines[thread_id];

    // each iteration saves output_width values T as one vec_type.
    // the number of vec_types produced by the loop is a multiple of threads_per_generator
    // since all threads_per_generator must contribute to produce the next values
    vec_type* vec_data = reinterpret_cast<vec_type*>(data + misalignment);
    size_t    index;
    for(index = thread_id; index < vec_n; index += stride)
    {
        for(unsigned int i = 0; i < input_width; i++)
        {
            input[i] = engine();
        }

        distribution(input, output);

        vec_data[index] = *reinterpret_cast<vec_type*>(output);
    }

    // deal with the non-aligned elements

    // number of elements T that come before and after the aligned elements
    const unsigned int remainder = tail_size + head_size;
    // number of output_width values T, rounded up
    // also round up to threads_per_generator ensure that all eight threads participate in the calculation
    const unsigned int remainder_ceil = (remainder + full_output_width - 1) / full_output_width;

    // each iteration saves at most output_width values T
    for(; index < vec_n + remainder_ceil; index += stride)
    {
        for(unsigned int i = 0; i < input_width; i++)
        {
            input[i] = engine();
        }

        distribution(input, output);

        for(unsigned int o = 0; o < output_width; o++)
        {
            // only write the elements that are still required, which means
            // that some random numbers get discarded
            unsigned int idx = output_width * index + o;
            if(o < output_width * vec_n + remainder)
            {
                // tail elements get wrapped around
                data[idx % size] = output[o];
            }
        }
    }

    // save state
    engines[thread_id] = engine;
}

} // end namespace detail
} // end namespace rocrand_host

class rocrand_mt19937 : public rocrand_generator_type<ROCRAND_RNG_PSEUDO_MT19937>
{
public:
    using base_type        = rocrand_generator_type<ROCRAND_RNG_PSEUDO_MT19937>;
    using octo_engine_type = ::rocrand_host::detail::mt19937_octo_engine;

    rocrand_mt19937(unsigned long long seed = 0, hipStream_t stream = 0)
        : base_type(seed, 0, stream), m_engines_initialized(false), m_engines(NULL)
    {
        // Allocate device random number engines
        auto error = hipMalloc(reinterpret_cast<void**>(&m_engines),
                               threads_per_generator * generator_count * sizeof(octo_engine_type));
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
    }

    rocrand_mt19937(const rocrand_mt19937&) = delete;

    rocrand_mt19937(rocrand_mt19937&&) = delete;

    rocrand_mt19937& operator=(const rocrand_mt19937&) = delete;

    rocrand_mt19937& operator=(rocrand_mt19937&&) = delete;

    ~rocrand_mt19937()
    {
        ROCRAND_HIP_FATAL_ASSERT(hipFree(m_engines));
    }

    void reset()
    {
        m_engines_initialized = false;
    }

    /// Changes seed to \p seed and resets generator state.
    void set_seed(unsigned long long seed)
    {
        m_seed                = seed;
        m_engines_initialized = false;
    }

    void set_order(rocrand_ordering order)
    {
        m_order               = order;
        m_engines_initialized = false;
    }

    rocrand_status init()
    {
        hipError_t err;

        if(m_engines_initialized)
        {
            return ROCRAND_STATUS_SUCCESS;
        }

        unsigned int* d_engines{};
        err = hipMalloc(reinterpret_cast<void**>(&d_engines),
                        generator_count * rocrand_host::detail::n * sizeof(unsigned int));
        if(err != hipSuccess)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        unsigned int* d_mt19937_jump{};
        err = hipMalloc(reinterpret_cast<void**>(&d_mt19937_jump), sizeof(rocrand_h_mt19937_jump));
        if(err != hipSuccess)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_engines));
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        err = hipMemcpy(d_mt19937_jump,
                        rocrand_h_mt19937_jump,
                        sizeof(rocrand_h_mt19937_jump),
                        hipMemcpyHostToDevice);
        if(err != hipSuccess)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_engines));
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_mt19937_jump));
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        hipLaunchKernelGGL(rocrand_host::detail::jump_ahead_kernel,
                           dim3(generator_count),
                           dim3(jump_ahead_thread_count),
                           0,
                           m_stream,
                           d_engines,
                           m_seed,
                           d_mt19937_jump);

        err = hipStreamSynchronize(m_stream);
        if(err != hipSuccess)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_engines));
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_mt19937_jump));
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }

        err = hipFree(d_mt19937_jump);
        if(err != hipSuccess)
        {
            hipFree(d_engines);
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        hipLaunchKernelGGL(rocrand_host::detail::init_engines_kernel,
                           dim3(block_count),
                           dim3(thread_count),
                           0,
                           m_stream,
                           m_engines,
                           d_engines);

        err = hipStreamSynchronize(m_stream);
        if(err != hipSuccess)
        {
            hipFree(d_engines);
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }

        err = hipFree(d_engines);
        if(err != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        m_engines_initialized = true;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T>>
    rocrand_status generate(T* data, size_t data_size, Distribution distribution = Distribution())
    {
        rocrand_status status = init();
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        hipLaunchKernelGGL(rocrand_host::detail::generate_kernel,
                           dim3(block_count),
                           dim3(thread_count),
                           0,
                           m_stream,
                           m_engines,
                           data,
                           data_size,
                           distribution);

        // check kernel status
        if(hipGetLastError() != hipSuccess)
        {
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T* data, size_t data_size)
    {
        uniform_distribution<T> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T* data, size_t data_size, T mean, T stddev)
    {
        normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T* data, size_t data_size, T mean, T stddev)
    {
        log_normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    rocrand_status generate_poisson(unsigned int* data, size_t data_size, double lambda)
    {
        try
        {
            m_poisson.set_lambda(lambda);
        }
        catch(rocrand_status status)
        {
            return status;
        }
        return generate(data, data_size, m_poisson.dis);
    }

private:
    bool              m_engines_initialized;
    octo_engine_type* m_engines;

    static constexpr unsigned int generators_per_block = thread_count / threads_per_generator;
    static constexpr unsigned int block_count          = generator_count / generators_per_block;
    static_assert(generator_count % generators_per_block == 0,
                  "generator count must be a multiple of generators per block");

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<> m_poisson;

    // m_seed from base_type
    // m_offset from base_type
};

#endif // ROCRAND_RNG_MT19937_H_
