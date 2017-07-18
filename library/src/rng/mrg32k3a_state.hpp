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

#ifndef ROCRAND_RNG_MRG32K3A_STATE_H_
#define ROCRAND_RNG_MRG32K3A_STATE_H_

#define ROCRAND_RNG_MRG32K3A_POW32 4294967296
#define ROCRAND_RNG_MRG32K3A_M1 4294967087
#define ROCRAND_RNG_MRG32K3A_M1C 209
#define ROCRAND_RNG_MRG32K3A_M2 4294944443
#define ROCRAND_RNG_MRG32K3A_M2C 22853
#define ROCRAND_RNG_MRG32K3A_A12 1403580
#define ROCRAND_RNG_MRG32K3A_A13 (4294967087 -  810728)
#define ROCRAND_RNG_MRG32K3A_A13N 810728
#define ROCRAND_RNG_MRG32K3A_A21 527612
#define ROCRAND_RNG_MRG32K3A_A23 (4294944443 - 1370589)
#define ROCRAND_RNG_MRG32K3A_A23N 1370589

// TODO: During optimisation stage, precompute all matrices for discard_sequence
// And try local reduction pattern

// TODO: Replace modulo operations in discard and discard_sequence

struct rocrand_mrg32k3a_state
{
    unsigned long long g1[3];
    unsigned long long g2[3];
    
    __host__ __device__
    rocrand_mrg32k3a_state()
    {
        g1[0] = 12345;
        g1[1] = 12345;
        g1[2] = 12345;
        g2[0] = 12345;
        g2[1] = 12345;
        g2[2] = 12345;
    }
    
    __host__ __device__
    rocrand_mrg32k3a_state(unsigned long long seed)
    {
        g1[0] = seed;
        g1[1] = seed;
        g1[2] = seed;
        g2[0] = seed;
        g2[1] = seed;
        g2[2] = seed;
    }
    
    __host__ __device__
    ~rocrand_mrg32k3a_state() {}

    
    inline __host__ __device__
    void discard(unsigned long long n)
    {
        unsigned long long A1[9] = 
        {
            0,                        1,   0,
            0,                        0,   1,
            ROCRAND_RNG_MRG32K3A_A13, ROCRAND_RNG_MRG32K3A_A12, 0
        };
        unsigned long long A2[9] = 
        {
            0,                        1, 0,
            0,                        0, 1,
            ROCRAND_RNG_MRG32K3A_A23, 0, ROCRAND_RNG_MRG32K3A_A21
        };
        
        while(n > 0) {
            if (n % 2 == 1) {
                mod_mat_vec(A1, g1, ROCRAND_RNG_MRG32K3A_M1);
                mod_mat_vec(A2, g2, ROCRAND_RNG_MRG32K3A_M2);
            }
            n = n / 2;

            mod_mat_sq(A1, ROCRAND_RNG_MRG32K3A_M1);
            mod_mat_sq(A2, ROCRAND_RNG_MRG32K3A_M2);
        }
    }
    
    inline __host__ __device__
    void discard()
    {
        discard(1);
    }

    inline __host__ __device__
    void discard_sequence(unsigned long long n)
    {
        unsigned long long A1p76[9] = 
        {
            82758667, 1871391091, 4127413238,
            3672831523, 69195019, 1871391091,
            3672091415, 3528743235, 69195019
        };
        unsigned long long A2p76[9] = 
        {
            1511326704, 3759209742, 1610795712,
            4292754251, 1511326704, 3889917532,
            3859662829, 4292754251, 3708466080
        };
        
        while(n > 0) {
            if (n % 2 == 1) {
                mod_mat_vec(A1p76, g1, ROCRAND_RNG_MRG32K3A_M1);
                mod_mat_vec(A2p76, g2, ROCRAND_RNG_MRG32K3A_M2);
            }
            n = n / 2;

            mod_mat_sq(A1p76, ROCRAND_RNG_MRG32K3A_M1);
            mod_mat_sq(A2p76, ROCRAND_RNG_MRG32K3A_M2);
        }
    }
    
    inline __device__ __host__
    void reset()
    {

    }

    inline __device__ __host__
    void set_seed(unsigned long long seed)
    {
        unsigned int x = (unsigned int) seed ^ 0x55555555UL;
        unsigned int y = (unsigned int) ((seed >> 32) ^ 0xAAAAAAAAUL);
        g1[0] = mod_mul_m1(x, seed);
        g1[1] = mod_mul_m1(y, seed);
        g1[2] = mod_mul_m1(x, seed);
        g2[0] = mod_mul_m2(y, seed);
        g2[1] = mod_mul_m2(x, seed);
        g2[2] = mod_mul_m2(y, seed);
        reset();
    }
    
    private:
        inline __device__ __host__
        void mod_mat_vec(unsigned long long * A, 
                         unsigned long long * s, 
                         unsigned long long m)
        {
            unsigned long long x[3];
            for (size_t i = 0; i < 3; ++i) {
                x[i] = 0;
                for (size_t j = 0; j < 3; j++)
                    x[i] = (A[i + 3 * j] * s[j] + x[i]) % m;
            }
            for (size_t i = 0; i < 3; ++i)
                s[i] = x[i];
        }
    
        inline __device__ __host__
        void mod_mat_sq(unsigned long long * A, 
                        unsigned long long m)
        {
            unsigned long long x[9];
            unsigned long long a;
            for (size_t i = 0; i < 3; i++) {
                for (size_t j = 0; j < 3; j++) {
                    a = 0;
                    for (size_t k = 0; k < 3; k++) {
                        a += (A[i + 3 * k] * A[k + 3 * j]) % m;
                    }
                    x[i + 3 * j] = a % m;
                }
            }
            for (size_t i = 0; i < 3; i++) {
                A[i + 3 * 0] = x[i + 3 * 0];
                A[i + 3 * 1] = x[i + 3 * 1];
                A[i + 3 * 2] = x[i + 3 * 2];
            }
        }
    
        inline __host__ __device__
        unsigned long long mod_mul_m1(unsigned int i, 
                                      unsigned long long j)
        {
            long long hi, lo, temp1, temp2;

            hi = i / 131072;
            lo = i - (hi * 131072);
            temp1 = mod_m1(hi * j) * 131072;
            temp2 = mod_m1(lo * j);
            lo = mod_m1(temp1 + temp2);

            if (lo < 0) 
                lo += ROCRAND_RNG_MRG32K3A_M1;
            return lo;
        }
    
        inline __host__ __device__
        unsigned long long mod_m1(unsigned long long i)
        {
            unsigned long long p;
            p = (i & (ROCRAND_RNG_MRG32K3A_POW32 - 1)) + (i >> 32) 
                * ROCRAND_RNG_MRG32K3A_M1C;
            if (p >= ROCRAND_RNG_MRG32K3A_M1) 
                p -= ROCRAND_RNG_MRG32K3A_M1;
            
            return p;
        }
    
        inline __host__ __device__
        unsigned long long mod_mul_m2(unsigned int i, 
                                      unsigned long long j)
        {
            long long hi, lo, temp1, temp2;

            hi = i / 131072;
            lo = i - (hi * 131072);
            temp1 = mod_m2(hi * j) * 131072;
            temp2 = mod_m2(lo * j);
            lo = mod_m2(temp1 + temp2);

            if (lo < 0) 
                lo += ROCRAND_RNG_MRG32K3A_M2;
            return lo;
        }
    
        inline __host__ __device__
        unsigned long long mod_m2(unsigned long long i)
        {
            unsigned long long p;
            p = (i & (ROCRAND_RNG_MRG32K3A_POW32 - 1)) + (i >> 32) 
                * ROCRAND_RNG_MRG32K3A_M2C;
            p = (p & (ROCRAND_RNG_MRG32K3A_POW32 - 1)) + (p >> 32) 
                * ROCRAND_RNG_MRG32K3A_M2C;
            if (p >= ROCRAND_RNG_MRG32K3A_M2) 
                p -= ROCRAND_RNG_MRG32K3A_M2;
            
            return p;
        }
};

#endif // ROCRAND_RNG_MRG32K3A_STATE_H_
