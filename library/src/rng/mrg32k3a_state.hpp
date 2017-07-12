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

#define PI 3.14159265358979323846f
#define NORM 2.3283065498378288e-10
#define POW32 4294967296
#define M1 4294967087
#define M1C 209
#define M2 4294944443
#define M2C 22853
#define A12 1403580
#define A13 (4294967087 -  810728)
#define A13N 810728
#define A21 527612
#define A23 (4294944443 - 1370589)
#define A23N 1370589

__device__
unsigned long long A1p76[3][3] = {
    {82758667, 1871391091, 4127413238},
    {3672831523, 69195019, 1871391091},
    {3672091415, 3528743235, 69195019}
};

__device__
unsigned long long A2p76[3][3] = {
    {1511326704, 3759209742, 1610795712},
    {4292754251, 1511326704, 3889917532},
    {3859662829, 4292754251, 3708466080}
};

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
    
    inline __host__ __device__
    void discard(unsigned long long n)
    {
        for (int i = 0; i < n; ++i) {
            modMatVec(A1p76, g1, g1, M1);
            modMatVec(A2p76, g2, g2, M2); 
        }
    }
    
    inline __host__ __device__
    void discard()
    {
        modMatVec(A1p76, g1, g1, M1);
        modMatVec(A2p76, g2, g2, M2); 
    }

    inline __host__ __device__
    void discard_sequence(unsigned long long n)
    {
        
    }
    
    inline __device__ __host__
    void reset()
    {

    }

    inline __device__ __host__
    void set_seed(unsigned long long seed)
    {
        g1[0] = seed;
        g1[1] = seed;
        g1[2] = seed;
        g2[0] = seed;
        g2[1] = seed;
        g2[2] = seed;
        reset();
    }
    
    private:
        inline __device__ __host__
        unsigned long long modMult(unsigned long long a, 
                                   unsigned long long s, 
                                   unsigned long long c, 
                                   long m)
        {   
            return (((unsigned long long) a * s + c) % m);
        }
    
        inline __device__ __host__
        void modMatVec (unsigned long long A[3][3], 
                        unsigned long long s[3], 
                        unsigned long long v[3], 
                        long m)
        {
            unsigned long long x[3];
            for (size_t i = 0; i < 3; ++i) {
                x[i] = 0;
                for (size_t j = 0; j < 3; j++)
                    x[i] = modMult(A[i][j], s[j], x[i], m);
                }
            for (size_t i = 0; i < 3; ++i)
                v[i] = x[i];
        }
};

#endif // ROCRAND_RNG_MRG32K3A_STATE_H_
