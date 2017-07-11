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
        unsigned long long A1[9], A2[9], A1b[9], A2b[9], a1, a2, v1b[3], v2b[3];
        int i, j, k;
        
        // Initialize discard matrices
        A1[0] = 0;   A1[3] = 1;   A1[6] = 0;
        A1[1] = 0;   A1[4] = 0;   A1[7] = 1;
        A1[2] = A13; A1[5] = A12; A1[8] = 0; 

        A2[0] = 0;   A2[3] = 1; A2[6] = 0;
        A2[1] = 0;   A2[4] = 0; A2[7] = 1;
        A2[2] = A23; A2[5] = 0; A2[8] = A21;
        
        while (n > 0) {
            if (n % 2 == 1) {
                for (i = 0; i < 3; ++i) {
                    v1b[i] = ((A1[i + 3 * 0] * g1[0]) % M1
                            + (A1[i + 3 * 1] * g1[1]) % M1
                            + (A1[i + 3 * 2] * g1[2]) % M1) % M1;
                    v2b[i] = ((A2[i + 3 * 0] * g2[0]) % M2
                            + (A2[i + 3 * 1] * g2[1]) % M2
                            + (A2[i + 3 * 2] * g2[2]) % M2) % M2;
                }
                for (i = 0; i < 3; ++i) {
                    g1[i] = v1b[i];
                    g2[i] = v2b[i];
                }
            }
            
            n = n / 2;
            
            for (i = 0; i < 3; ++i) {
                for (j = 0; j < 3; ++j) {
                    a1 = 0;
                    a2 = 0;
                    for (k = 0; k < 3; ++k) {
                        a1 += (A1[i + 3 * k] * A1[k + 3 * j]) % M1;
                        a2 += (A2[i + 3 * k] * A2[k + 3 * j]) % M2;
                    }
                    A1b[i + 3 * j] = a1 % M1;
                    A2b[i + 3 * j] = a2 % M2;
                }
            }
            
            for (i = 0; i < 9; ++i) {
                A1[i] = A1b[i];
                A2[i] = A2b[i];
            }
        }
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
};

#endif // ROCRAND_RNG_MRG32K3A_STATE_H_
