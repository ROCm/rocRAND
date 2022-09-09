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
#define MT_FQUALIFIERS_HOST __host__

#include <hip/hip_runtime.h>

#include <rocrand/rocrand.h>

#include "common.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"
#include "mt19937_precomputed.hpp"

namespace
{
/// Number of independent generators. Value is fixed to produce deterministic number stream.
static constexpr unsigned int generator_count = 8192U;
/// Number of threads that cooporate to run one generator. Value is fixed in implementation.
static constexpr unsigned int threads_per_generator = 8U;
/// Number of threads per block. Can be tweaked for performance.
static constexpr unsigned int thread_count = 256U;
static_assert(thread_count % threads_per_generator == 0U,
              "all eight threads of the generator must be in the same block");
} // namespace

namespace rocrand_host
{
namespace detail
{

namespace
{
// MT19937 constants.

/// Number of elements in the state vector.
static constexpr unsigned int n = 624U;
/// The next value of element \p i depends on <tt>i + 1 % n<\tt> and <tt>i + m % n<\tt>.
static constexpr unsigned int m = 397U;
/// Constant vector A.
static constexpr unsigned int matrix_a = 0x9908B0DFU;
/// Most significant <tt>w - r<\tt> bits.
static constexpr unsigned int upper_mask = 0x80000000U;
/// Least significant \p r bits.
static constexpr unsigned int lower_mask = 0x7FFFFFFFU;
} // namespace

struct mt19937_engine
{
    // Jumping constants.

    static constexpr unsigned int qq = 7;
    /// <tt>ll = 2 ^ qq<\tt>.
    static constexpr unsigned int ll = 128;

    struct mt19937_state
    {
        unsigned int mt[n];
        // index of the next value to be calculated
        unsigned int ptr;
    };

    MT_FQUALIFIERS_HOST mt19937_engine(unsigned long long seed)
    {
        const unsigned int seedu = (seed >> 32) ^ seed;
        m_state.mt[0]            = seedu;
        for(unsigned int i = 1; i < n; i++)
        {
            m_state.mt[i] = 1812433253U * (m_state.mt[i - 1] ^ (m_state.mt[i - 1] >> 30)) + i;
        }
        m_state.ptr = 0;
    }

    /// Advances the internal state to skip a single subsequence, which is <tt>2 ^ 1000</tt> states long.
    MT_FQUALIFIERS_HOST void discard_subsequence()
    {
        m_state = discard_subsequence_impl(mt19937_jump, m_state);
    }

    // Generates the next state.
    static MT_FQUALIFIERS_HOST void gen_next(mt19937_state& state)
    {
        /// mag01[x] = x * matrix_a for x in [0, 1]
        constexpr unsigned int mag01[2] = {0x0U, matrix_a};
        int                    num;
        unsigned int           y;
        int                    tmp_n = n;
        int                    tmp_m = m;

        num = state.ptr;
        if(num < tmp_n - tmp_m)
        {
            y             = (state.mt[num] & upper_mask) | (state.mt[num + 1] & lower_mask);
            state.mt[num] = state.mt[num + tmp_m] ^ (y >> 1) ^ mag01[y & 0x1U];
            state.ptr++;
        }
        else if(num < tmp_n - 1)
        {
            y             = (state.mt[num] & upper_mask) | (state.mt[num + 1] & lower_mask);
            state.mt[num] = state.mt[num + (tmp_m - tmp_n)] ^ (y >> 1) ^ mag01[y & 0x1U];
            state.ptr++;
        }
        else if(num == tmp_n - 1)
        {
            y                   = (state.mt[tmp_n - 1] & upper_mask) | (state.mt[0] & lower_mask);
            state.mt[tmp_n - 1] = state.mt[tmp_m - 1] ^ (y >> 1) ^ mag01[y & 0x1U];
            state.ptr           = 0;
        }
    }

    /// Return the i-th coefficient of the polynomial pf.
    static MT_FQUALIFIERS_HOST unsigned long get_coef(const unsigned int pf[mt19937_p_size],
                                                      unsigned int       deg)
    {
        constexpr unsigned int log_w_size  = 5U;
        constexpr unsigned int w_size_mask = 0x1FU;
        return (pf[deg >> log_w_size] & (mt19937_lsb << (deg & w_size_mask))) != 0;
    }

    /// Copy state \p ss into state <tt>ts</tt>.
    static MT_FQUALIFIERS_HOST void copy_state(mt19937_state& ts, const mt19937_state& ss)
    {
        for(size_t i = 0; i < n; i++)
        {
            ts.mt[i] = ss.mt[i];
        }

        ts.ptr = ss.ptr;
    }

    /// Add state \p s2 to state <tt>s1</tt>.
    static MT_FQUALIFIERS_HOST void add_state(mt19937_state& s1, const mt19937_state& s2)
    {
        int i, pt1 = s1.ptr, pt2 = s2.ptr;
        int tmp_n = n;

        if(pt2 - pt1 >= 0)
        {
            for(i = 0; i < tmp_n - pt2; i++)
            {
                s1.mt[i + pt1] ^= s2.mt[i + pt2];
            }
            for(; i < tmp_n - pt1; i++)
            {
                s1.mt[i + pt1] ^= s2.mt[i + (pt2 - tmp_n)];
            }
            for(; i < tmp_n; i++)
            {
                s1.mt[i + (pt1 - tmp_n)] ^= s2.mt[i + (pt2 - tmp_n)];
            }
        }
        else
        {
            for(i = 0; i < tmp_n - pt1; i++)
            {
                s1.mt[i + pt1] ^= s2.mt[i + pt2];
            }
            for(; i < tmp_n - pt2; i++)
            {
                s1.mt[i + (pt1 - tmp_n)] ^= s2.mt[i + pt2];
            }
            for(; i < tmp_n; i++)
            {
                s1.mt[i + (pt1 - tmp_n)] ^= s2.mt[i + (pt2 - tmp_n)];
            }
        }
    }

    /// Generate Gray code.
    static MT_FQUALIFIERS_HOST void gray_code(unsigned int h[ll])
    {
        h[0] = 0;

        unsigned int l    = 1;
        unsigned int term = ll;
        unsigned int j    = 1;
        for(unsigned int i = 1; i <= qq; i++)
        {
            l    = (l << 1);
            term = (term >> 1);
            for(; j < l; j++)
            {
                h[j] = h[l - j - 1] ^ term;
            }
        }
    }

    /// Compute h(f)ss where h(t) are exact q-degree polynomials,
    /// f is the transition function, and ss is the initial state
    /// the results are stored in vec_h[0] , ... , vec_h[ll-1].
    static MT_FQUALIFIERS_HOST void gen_vec_h(const mt19937_state& ss, mt19937_state vec_h[ll])
    {
        mt19937_state v{};
        unsigned int  h[ll];

        gray_code(h);

        copy_state(vec_h[0], ss);

        for(unsigned int i = 0; i < qq; i++)
        {
            gen_next(vec_h[0]);
        }

        for(unsigned int i = 1; i < ll; i++)
        {
            copy_state(v, ss);
            unsigned int g = h[i] ^ h[i - 1];
            for(unsigned int k = 1; k < g; k = (k << 1))
            {
                gen_next(v);
            }
            copy_state(vec_h[h[i]], vec_h[h[i - 1]]);
            add_state(vec_h[h[i]], v);
        }
    }

    /// Compute pf(ss) using Sliding window algorithm.
    static MT_FQUALIFIERS_HOST mt19937_state calc_state(const unsigned int   pf[mt19937_p_size],
                                                        const mt19937_state& ss,
                                                        const mt19937_state  vec_h[ll])
    {
        mt19937_state tmp{};
        int           i = mt19937_mexp - 1;

        while(get_coef(pf, i) == 0)
        {
            i--;
        }

        for(; i >= static_cast<int>(qq); i--)
        {
            if(get_coef(pf, i) != 0)
            {
                for(int j = 0; j < static_cast<int>(qq) + 1; j++)
                {
                    gen_next(tmp);
                }
                int digit = 0;
                for(int j = 0; j < static_cast<int>(qq); j++)
                {
                    digit = (digit << 1) ^ get_coef(pf, i - j - 1);
                }
                add_state(tmp, vec_h[digit]);
                i -= qq;
            }
            else
            {
                gen_next(tmp);
            }
        }

        for(; i > -1; i--)
        {
            gen_next(tmp);
            if(get_coef(pf, i) == 1)
            {
                add_state(tmp, ss);
            }
        }

        return tmp;
    }

    /// Computes jumping ahead with Sliding window algorithm.
    static MT_FQUALIFIERS_HOST mt19937_state
        discard_subsequence_impl(const unsigned int pf[mt19937_p_size], const mt19937_state& ss)
    {
        // skip state
        mt19937_state vec_h[ll];
        gen_vec_h(ss, vec_h);
        mt19937_state new_state = calc_state(pf, ss, vec_h);

        // rotate the array to align ptr with the array boundary
        if(new_state.ptr != 0)
        {
            unsigned int tmp[n];
            for(unsigned int i = 0; i < n; i++)
            {
                tmp[i] = new_state.mt[(i + new_state.ptr) % n];
            }

            for(unsigned int i = 0; i < n; i++)
            {
                new_state.mt[i] = tmp[i];
            }
        }

        // set to 0, which is the index of the next number to be calculated
        new_state.ptr = 0;

        return new_state;
    }

public:
    mt19937_state m_state;
};

struct mt19937_octo_engine
{
    /// Tuples of \p items_per_thread follow a regular pattern.
    static constexpr unsigned int items_per_thread = 7U;

    struct mt19937_octo_state
    {
        /// Thread 0 has element   0, thread 1 has element 113, thread 2 has element 170,
        /// thread 3 had element 283, thread 4 has element 340, thread 5 has element 397,
        /// thread 6 has element 510, thread 7 has element 567.
        /// Thread i for i in [0, 7), has the following elements:
        /// [  1 + ipt * i,   1 + ipt * (i + 1)), [398 + ipt * i, 398 + ipt * (i + 1)), [171 + ipt * i, 171 + ipt * (i + 1)),
        /// [568 + ipt * i, 568 + ipt * (i + 1)), [341 + ipt * i, 341 + ipt * (i + 1)), [114 + ipt * i, 114 + ipt * (i + 1)),
        /// [511 + ipt * i, 511 + ipt * (i + 1)), [284 + ipt * i, 284 + ipt * (i + 1)), [ 57 + ipt * i,  57 + ipt * (i + 1)),
        /// [454 + ipt * i, 454 + ipt * (i + 1)), [227 + ipt * i, 227 + ipt * (i + 1))
        ///
        /// which are 1 + 11 * 7 = 77 values per thread.
        unsigned int mt[1U + items_per_thread * 11U];
        /// The index of the next value to be returned in global array of values.
        /// The actual returned values are of a different order per n elements.
        /// When \p mti is <tt>n</tt>, \p n new values must be calculated.
        unsigned int mti;
    };

    /// Constants to map the indices to \p mt19937_octo_state.mt_extra indices.
    /// For example, \p i000_0 is the index of \p 0 owned by thread 0.

    static constexpr unsigned int i000_0 = 0U;
    static constexpr unsigned int i113_1 = 0U;
    static constexpr unsigned int i170_2 = 0U;
    static constexpr unsigned int i283_3 = 0U;
    static constexpr unsigned int i340_4 = 0U;
    static constexpr unsigned int i397_5 = 0U;
    static constexpr unsigned int i510_6 = 0U;
    static constexpr unsigned int i567_7 = 0U;

    /// Constants used to map the indices to \p mt19937_octo_state.mt indices.
    /// For example, \p i001 is the index of <tt>1 + tid * ipt</tt>.

    static constexpr unsigned int i001 = 1U + items_per_thread * 0U;
    static constexpr unsigned int i057 = 1U + items_per_thread * 1U;
    static constexpr unsigned int i114 = 1U + items_per_thread * 2U;
    static constexpr unsigned int i171 = 1U + items_per_thread * 3U;
    static constexpr unsigned int i227 = 1U + items_per_thread * 4U;
    static constexpr unsigned int i284 = 1U + items_per_thread * 5U;
    static constexpr unsigned int i341 = 1U + items_per_thread * 6U;
    static constexpr unsigned int i398 = 1U + items_per_thread * 7U;
    static constexpr unsigned int i454 = 1U + items_per_thread * 8U;
    static constexpr unsigned int i511 = 1U + items_per_thread * 9U;
    static constexpr unsigned int i568 = 1U + items_per_thread * 10U;

    /// Initialize the octo engine from the engine it shares with seven other threads.
    MT_FQUALIFIERS void gather(const mt19937_engine* engine)
    {
        constexpr unsigned int off_cnt = 11U;
        /// Used to map the \p mt19937_octo_state.mt indices to \p mt19937_state.mt indices.
        static constexpr unsigned int offsets[off_cnt]
            = {1U, 57U, 114U, 171U, 227U, 284U, 341U, 398U, 454U, 511U, 568U};

        const unsigned int tid = threadIdx.x & 7U;

        // initialize the elements that follow a regular pattern
        for(unsigned int i = 0; i < off_cnt; i++)
        {
            for(unsigned int j = 0; j < items_per_thread; j++)
            {
                const unsigned int index = offsets[i] + items_per_thread * tid + j;
                // + 1 for the special value
                m_state.mt[1U + items_per_thread * i + j] = engine->m_state.mt[index];
            }
        }

        // initialize the elements that do not follow a regular pattern
        constexpr unsigned int dest_idx[threads_per_generator]
            = {i000_0, i113_1, i170_2, i283_3, i340_4, i397_5, i510_6, i567_7};
        constexpr unsigned int src_idx[threads_per_generator]
            = {0, 113, 170, 283, 340, 397, 510, 567};
        m_state.mt[dest_idx[tid]] = engine->m_state.mt[src_idx[tid]];

        // set to n, to indicate that a batch of n values can be calculated at a time
        m_state.mti = n;
    }

    /// Returns \p val from thread <tt>tid mod 8</tt>.
    static MT_FQUALIFIERS unsigned int shuffle(unsigned int val, unsigned int tid)
    {
        return __shfl(val, tid, 8U);
    }

    /// For thread i, returns \p val from thread <tt>(i + 1) mod 8</tt>
    static MT_FQUALIFIERS unsigned int shuffle_down(unsigned int val)
    {
        return __shfl_down(val, 1U, 8U);
    }

    /// For thread i, returns \p val from thread <tt>(i - 1) mod 8</tt>
    static MT_FQUALIFIERS unsigned int shuffle_up(unsigned int val)
    {
        return __shfl_up(val, 1U, 8U);
    }
    /// Calculates value of index \p i using values <tt>i</tt>, <tt>(i + 1) % n</tt>, and <tt>(i + m) % n</tt>.
    static MT_FQUALIFIERS unsigned int
        comp(unsigned int mt_i, unsigned int mt_i_1, unsigned int mt_i_m)
    {
        const unsigned int y = (mt_i & upper_mask) | (mt_i_1 & lower_mask);
        /// mag01[x] = x * matrix_a for x in [0, 1]
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

            // extract the first iteration from the loop
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

ROCRAND_KERNEL
__launch_bounds__(thread_count) void init_engines_kernel(mt19937_octo_engine* octo_engines,
                                                         mt19937_engine*      engines)
{
    const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    // every eight octo engines gather from the same engine
    octo_engines[thread_id].gather(&engines[thread_id / threads_per_generator]);
}

template<class T, class Distribution>
ROCRAND_KERNEL __launch_bounds__(thread_count) void generate_kernel(mt19937_octo_engine* engines,
                                                                    T*                   data,
                                                                    const size_t         size,
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
    size_t    index    = thread_id;
    while(index < vec_n)
    {
        for(unsigned int i = 0; i < input_width; i++)
        {
            input[i] = engine();
        }

        distribution(input, output);

        vec_data[index] = *reinterpret_cast<vec_type*>(output);

        // next position
        index += stride;
    }

    // deal with the non-aligned elements

    // number of elements T that come before and after the aligned elements
    const unsigned int remainder = tail_size + head_size;
    // number of output_width values T, rounded up
    // also round up to threads_per_generator ensure that all eight threads participate in the calculation
    const unsigned int remainder_ceil = (remainder + full_output_width - 1) / full_output_width;

    // each iteration saves at most output_width values T
    while(index < vec_n + remainder_ceil)
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

        index += stride;
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
    using engine_type      = ::rocrand_host::detail::mt19937_engine;

    rocrand_mt19937(unsigned long long seed = 0, hipStream_t stream = 0)
        : base_type(seed, 0, stream), m_engines_initialized(false), m_engines(NULL)
    {
        // Allocate device random number engines
        auto error = hipMalloc(&m_engines,
                               threads_per_generator * generator_count * sizeof(octo_engine_type));
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
    }

    ~rocrand_mt19937()
    {
        hipFree(m_engines);
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

    rocrand_status init()
    {
        hipError_t err;

        if(m_engines_initialized)
        {
            return ROCRAND_STATUS_SUCCESS;
        }

        // initialize the engines on the host due to high memory requirement
        // for jumping subsequences
        engine_type* h_engines
            = static_cast<engine_type*>(malloc(generator_count * sizeof(engine_type)));
        h_engines[0] = engine_type(m_seed);
        for(size_t i = 1; i < generator_count; i++)
        {
            // every consecutive engine is one subsequence away from the previous
            h_engines[i] = h_engines[i - 1];
            h_engines[i].discard_subsequence();
        }

        engine_type* d_engines{};
        err = hipMalloc(&d_engines, generator_count * sizeof(engine_type));
        if(err != hipSuccess)
        {
            free(h_engines);
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        err = hipMemcpy(d_engines,
                        h_engines,
                        generator_count * sizeof(engine_type),
                        hipMemcpyHostToDevice);

        free(h_engines);

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
