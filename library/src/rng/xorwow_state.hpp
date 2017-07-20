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

#ifndef ROCRAND_RNG_XORWOW_STATE_H_
#define ROCRAND_RNG_XORWOW_STATE_H_

#include <rocrand_xorwow_precomputed.h>

// G. Marsaglia, Xorshift RNGs, 2003
// http://www.jstatsoft.org/v08/i14/paper

namespace rocrand_xorwow_state_detail
{

inline __host__ __device__
void copy_mat(unsigned int * dst, const unsigned int * src)
{
    for (int i = 0; i < XORWOW_SIZE; i++)
    {
        dst[i] = src[i];
    }
}

inline __host__ __device__
void copy_vec(unsigned int * dst, const unsigned int * src)
{
    for (int i = 0; i < XORWOW_N; i++)
    {
        dst[i] = src[i];
    }
}

inline __host__ __device__
void mul_mat_vec_inplace(const unsigned int * m, unsigned int * v)
{
    unsigned int r[XORWOW_N] = { 0 };
    for (int i = 0; i < XORWOW_N; i++)
    {
        for (int j = 0; j < XORWOW_M; j++)
        {
            if (v[i] & (1 << j))
            {
                for (int k = 0; k < XORWOW_N; k++)
                {
                    r[k] ^= m[XORWOW_N * (i * XORWOW_M + j) + k];
                }
            }
        }
    }
    copy_vec(v, r);
}

inline __host__ __device__
void mul_mat_mat_inplace(unsigned int * a, const unsigned int * b)
{
    for (int i = 0; i < XORWOW_N * XORWOW_M; i++)
    {
        mul_mat_vec_inplace(b, a + i * XORWOW_N);
    }
}

} // namespace rocrand_xorwow_state_detail

struct rocrand_xorwow_state
{
    // Xorshift values (160 bits)
    unsigned int x[5];

    // Weyl sequence value
    unsigned int d;

    __host__ __device__
    rocrand_xorwow_state()
        : rocrand_xorwow_state(0) {}

    __host__ __device__
    rocrand_xorwow_state(unsigned long long seed)
    {
        set_seed(seed);
    }

    __host__ __device__
    ~rocrand_xorwow_state() {}

    __forceinline__ __host__ __device__
    unsigned int operator()()
    {
        return d + x[4];
    }

    inline __host__ __device__
    void discard()
    {
        const unsigned int t = x[0] ^ (x[0] >> 2);
        x[0] = x[1];
        x[1] = x[2];
        x[2] = x[3];
        x[3] = x[4];
        x[4] = (x[4] ^ (x[4] << 4)) ^ (t ^ (t << 1));

        d += 362437;
    }

    inline __host__ __device__
    void discard(unsigned long long n)
    {
        #ifdef __HIP_DEVICE_COMPILE__
        jump(n, d_xorwow_jump_matrices);
        #else
        jump(n, h_xorwow_jump_matrices);
        #endif

        // Apply n steps to Weyl sequence value as well
        d += static_cast<unsigned int>(n) * 362437;
    }

    inline __host__ __device__
    void discard_sequence(unsigned long long n)
    {
        // Discard n * 2^67 samples
        #ifdef __HIP_DEVICE_COMPILE__
        jump(n, d_xorwow_sequence_jump_matrices);
        #else
        jump(n, h_xorwow_sequence_jump_matrices);
        #endif

        // d has the same value because 2^67 is divisible by 2^32 (d is 32-bit)
    }

    inline __host__ __device__
    void set_seed(unsigned long long seed)
    {
        x[0] = 123456789UL;
        x[1] = 362436069UL;
        x[2] = 521288629UL;
        x[3] = 88675123UL;
        x[4] = 5783321UL;

        d = 6615241UL;

        // TODO all constants below are from cuRAND
        const unsigned int s0 = static_cast<unsigned int>(seed) ^ 0xaad26b49UL;
        const unsigned int s1 = static_cast<unsigned int>(seed >> 32) ^ 0xf7dcefddUL;
        const unsigned int t0 = 1099087573UL * s0;
        const unsigned int t1 = 2591861531UL * s1;
        x[0] = x[0] + t0;
        x[1] = x[1] ^ t0;
        x[2] = x[2] + t1;
        x[3] = x[3] ^ t1;
        x[4] = x[4] + t0;
        d = d + t1 + t0;
    }

private:

    inline __host__ __device__
    void jump(unsigned long long v, const unsigned int jump_matrices[XORWOW_JUMP_MATRICES][XORWOW_SIZE])
    {
        namespace detail = rocrand_xorwow_state_detail;

        // x~(n + v) = (A^v mod m)x~n mod m
        // The matrix (A^v mod m) can be precomputed for selected values of v.
        //
        // For XORWOW_JUMP_LOG2 = 2
        // xorwow_jump_matrices contains precomputed matrices:
        //   A^1, A^4, A^16...
        //
        // For XORWOW_JUMP_LOG2 = 2 and XORWOW_SEQUENCE_JUMP_LOG2 = 67
        // xorwow_sequence_jump_matrices contains precomputed matrices:
        //   A^(1 * 2^67), A^(4 * 2^67), A^(16 * 2^67)...
        //
        // Intermediate powers can calculated as multiplication of the powers above.
        // Powers after the last precomputed matrix can be calculated using
        // Exponentiation by squaring method.

        int mi = 0;
        while (v > 0 && mi < XORWOW_JUMP_MATRICES)
        {
            const int l = ((mi < XORWOW_JUMP_MATRICES - 1) ? XORWOW_JUMP_LOG2 : 1);
            for (int i = 0; i < (v & ((1 << l) - 1)); i++)
            {
                detail::mul_mat_vec_inplace(jump_matrices[mi], x);
            }
            mi++;
            v >>= l;
        }

        if (v > 0)
        {
            // All precomputed matrices are used, we need to use the last one
            // to create matrices of next powers of 2

#if defined(__HIP_PLATFORM_HCC__) && defined(__HIP_DEVICE_COMPILE__)
            // HCC kernel has performance issues because of spilled registers
            // for a and b.
            // Linear version is used here instead of matrix squaring.
            // This means that large offsets will need a lot of time to process.

            for (v = v << 1; v > 0; v--)
            {
                detail::mul_mat_vec_inplace(jump_matrices[XORWOW_JUMP_MATRICES - 1], x);
            }
#else // NVCC and host code
            unsigned int a[XORWOW_SIZE];
            unsigned int b[XORWOW_SIZE];

            detail::copy_mat(a, jump_matrices[XORWOW_JUMP_MATRICES - 1]);
            detail::copy_mat(b, a);

            // Exponentiation by squaring
            do
            {
                // Square the matrix
                detail::mul_mat_mat_inplace(a, b);
                detail::copy_mat(b, a);

                if (v & 1)
                {
                    detail::mul_mat_vec_inplace(b, x);
                }

                v >>= 1;
            } while (v > 0);
#endif
        }
    }
};

#endif // ROCRAND_RNG_XORWOW_STATE_H_
