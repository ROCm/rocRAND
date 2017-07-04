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

#ifndef ROCRAND_RNG_DISTRIBUTION_NORMAL_COMMON_H_
#define ROCRAND_RNG_DISTRIBUTION_NORMAL_COMMON_H_

#include "common.hpp"

inline __host__ __device__ float2 operator+(float b, float2 a)
{
    return make_float2(a.x + b, a.y + b);
}
    
inline __host__ __device__ float2 operator*(float b, float2 a)
{
    return make_float2(a.x * b, a.y * b);
}

inline __host__ __device__ float2 expf(float2 a)
{
    return make_float2(expf(a.x), expf(a.y));
}

inline __host__ __device__ float4 make_float4(float2 a, float2 b)
{
    return make_float4(a.x, a.y, b.x, b.y);
}
    
inline __host__ __device__ double2 operator+(double b, double2 a)
{
    return make_double2(a.x + b, a.y + b);
}
    
inline __host__ __device__ double2 operator*(double b, double2 a)
{
    return make_double2(a.x * b, a.y * b);
}

inline __host__ __device__ double2 expf(double2 a)
{
    return make_double2(exp(a.x), exp(a.y));
}

__host__ __device__ float2 box_muller(unsigned int x, unsigned int y)
{
    float2 result;
    float u = nextafter(x * ROC_2POW32_INV, 1.0f);
    float v = nextafter(y * ROC_2POW32_INV_2PI, ROC_2PI);
    float s = sqrtf(-2.0f * logf(u));
    #ifdef __HIP_DEVICE_COMPILE__
        __sincosf(v, &result.x, &result.y);
        result.x *= s;
        result.y *= s; 
    #else
        result.x = sinf(v) * s;
        result.y = cosf(v) * s;
    #endif
    return result;
}

__host__ __device__ double2 box_muller_double(uint4 xy)
{
    double2 result;
    unsigned long long zx = (unsigned long long)xy.x ^ 
        ((unsigned long long)xy.y << (53 - 32));
    double u = nextafter(zx * ROC_2POW53_INV_DOUBLE, 1.0);
    unsigned long long zy = (unsigned long long)xy.z ^ 
        ((unsigned long long)xy.w << (53 - 32));
    double v = nextafter(zy * (ROC_2POW53_INV_DOUBLE * 2.0), 2.0);
    double s = sqrt(-2.0 * log(u));
    #ifdef __HIP_DEVICE_COMPILE__
        sincospi(v, &result.x, &result.y);
        result.x *= s;
        result.y *= s;
    #else
        result.x = sin(v * ROC_PI_DOUBLE) * s;
        result.y = cos(v * ROC_PI_DOUBLE) * s;
    #endif
    return result;
}

// TODO: Improve implementation
__host__ __device__ float2 marsaglia(unsigned int x, unsigned int y)
{
    float2 result;
    float u = nextafter(x * ROC_2POW32_INV, 1.0f) * 2.0f - 1.0f;
    float v = nextafter(y * ROC_2POW32_INV, 1.0f) * 2.0f - 1.0f;
    float s = u * u + v * v;
    float multiplier = sqrtf(-2.0f * logf(s) / s);
    result.x = u * s;
    result.y = v * s;
    return result;
}

// TODO: Improve implementation
__host__ __device__ double2 marsaglia_double(uint4 xy)
{
    double2 result;
    unsigned long long zx = (unsigned long long)xy.x ^ 
        ((unsigned long long)xy.y << (53 - 32));
    double u = nextafter(zx * ROC_2POW53_INV_DOUBLE, 1.0) * 2.0 - 1.0;
    unsigned long long zy = (unsigned long long)xy.z ^ 
        ((unsigned long long)xy.w << (53 - 32));
    double v = nextafter(zy * ROC_2POW53_INV_DOUBLE, 1.0) * 2.0 - 1.0;
    double s = u * u + v * v;
    double multiplier = sqrt(-2.0f * log(s) / s);
    result.x = u * s;
    result.y = v * s;
    return result;
}

#endif // ROCRAND_RNG_DISTRIBUTION_NORMAL_COMMON_H_
