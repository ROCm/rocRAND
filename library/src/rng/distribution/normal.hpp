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

#ifndef ROCRAND_RNG_DISTRIBUTION_NORMAL_H_
#define ROCRAND_RNG_DISTRIBUTION_NORMAL_H_

#define ROC_2POW32_INV (2.3283064e-10f)
#define ROC_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
#define ROC_2POW53_INV_DOUBLE (1.1102230246251565e-16)
#define ROC_PI_DOUBLE  (3.1415926535897932)

template<class T>
struct normal_distribution;

__host__ __device__ float2 box_muller(unsigned int x, unsigned int y)
{
    float2 result;
    float u = x * ROC_2POW32_INV + (ROC_2POW32_INV / 2);
    float v = y * ROC_2POW32_INV_2PI + (ROC_2POW32_INV_2PI / 2);
    float s = sqrtf(-2.0f * logf(u));
    result.x = sinf(v);
    result.y = cosf(v);
    result.x *= s;
    result.y *= s; 
    return result;
}

__host__ __device__ double2 box_muller_double(unsigned int x0, unsigned int x1, 
                                             unsigned int y0, unsigned int y1)
{
    double2 result;
    unsigned long long zx = (unsigned long long)x0 ^ 
        ((unsigned long long)x1 << (53 - 32));
    double u = zx * ROC_2POW53_INV_DOUBLE + (ROC_2POW53_INV_DOUBLE / 2.0);
    unsigned long long zy = (unsigned long long)y0 ^ 
        ((unsigned long long)y1 << (53 - 32));
    double v = zy * (ROC_2POW53_INV_DOUBLE * 2.0) + ROC_2POW53_INV_DOUBLE;
    double s = sqrt(-2.0 * log(u));
    result.x = sin(v * ROC_PI_DOUBLE);
    result.y = cos(v * ROC_PI_DOUBLE);
    result.x *= s;
    result.y *= s;

    return result;
}

template<>
struct normal_distribution<float>
{
    __host__ __device__ float2 operator()(unsigned int x, unsigned int y)
    {
        float2 v = box_muller(x, y);
        return v;
    }

    __host__ __device__ float4 operator()(uint4 x)
    {
        float2 v = box_muller(x.x, x.y);
        float2 w = box_muller(x.z, x.w);
        float4 u = { v.x, v.y, w.x, w.y };
        return u;
    }
};

template<>
struct normal_distribution<double>
{
    __host__ __device__ double2 operator()(uint4 x)
    {
        double2 v = box_muller_double(x.x, x.y, x.z, x.w);
        return v;
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_NORMAL_H_
