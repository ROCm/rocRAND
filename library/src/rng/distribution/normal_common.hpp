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

#include <cmath>
#include <hip/hip_runtime.h>

#include "common.hpp"

__forceinline__ __host__ __device__
float2 box_muller(unsigned int x, unsigned int y)
{
    float2 result;
    float u = ROCRAND_2POW32_INV + (x * ROCRAND_2POW32_INV);
    float v = ROCRAND_2POW32_INV_2PI + (y * ROCRAND_2POW32_INV_2PI);
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

__forceinline__ __host__ __device__
double2 box_muller_double(uint4 xy)
{
    double2 result;
    unsigned long long zx = (unsigned long long)xy.x ^
        ((unsigned long long)xy.y << (53 - 32));
    double u = ROCRAND_2POW53_INV_DOUBLE + (zx * ROCRAND_2POW53_INV_DOUBLE);
    unsigned long long zy = (unsigned long long)xy.z ^
        ((unsigned long long)xy.w << (53 - 32));
    double v = (ROCRAND_2POW53_INV_DOUBLE * 2.0) + (zy * (ROCRAND_2POW53_INV_DOUBLE * 2.0));
    double s = sqrt(-2.0 * log(u));
    #ifdef __HIP_DEVICE_COMPILE__
        sincospi(v, &result.x, &result.y);
        result.x *= s;
        result.y *= s;
    #else
        result.x = sin(v * ROCRAND_PI_DOUBLE) * s;
        result.y = cos(v * ROCRAND_PI_DOUBLE) * s;
    #endif
    return result;
}

__forceinline__ __host__ __device__
float2 box_muller_mrg(float x, float y)
{
    float2 result;
    float u = x;
    float v = y * ROCRAND_2PI;
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

__forceinline__ __host__ __device__
double2 box_muller_double_mrg(double x, double y)
{
    double2 result;
    double u = x;
    double v = y * 2.0;
    double s = sqrt(-2.0 * log(u));
    #ifdef __HIP_DEVICE_COMPILE__
        sincospi(v, &result.x, &result.y);
        result.x *= s;
        result.y *= s;
    #else
        result.x = sin(v * ROCRAND_PI_DOUBLE) * s;
        result.y = cos(v * ROCRAND_PI_DOUBLE) * s;
    #endif
    return result;
}

__forceinline__ __host__ __device__
float roc_f_erfinv(float x)
{
    float tt1, tt2, lnx, sgn;
    sgn = (x < 0.0f) ? -1.0f : 1.0f;

    x = (1.0f - x) * (1.0f + x);
    lnx = logf(x);

    #ifdef __HIP_DEVICE_COMPILE__
    if (isnan(lnx))
    #else
    if (std::isnan(lnx))
    #endif
        return 1.0f;
    #ifdef __HIP_DEVICE_COMPILE__
    else if (isinf(lnx))
    #else
    else if (std::isinf(lnx))
    #endif
        return 0.0f;

    tt1 = 2.0f / (ROCRAND_PI * 0.147f) + 0.5f * lnx;
    tt2 = 1.0f / (0.147f) * lnx;

    return(sgn * sqrtf(-tt1 + sqrtf(tt1 * tt1 - tt2)));
}

__forceinline__ __host__ __device__
double roc_d_erfinv(double x)
{
    double tt1, tt2, lnx, sgn;
    sgn = (x < 0.0) ? -1.0 : 1.0;

    x = (1.0 - x) * (1.0 + x);
    lnx = log(x);

    #ifdef __HIP_DEVICE_COMPILE__
    if (isnan(lnx))
    #else
    if (std::isnan(lnx))
    #endif
        return 1.0;
    #ifdef __HIP_DEVICE_COMPILE__
    else if (isinf(lnx))
    #else
    else if (std::isinf(lnx))
    #endif
        return 0.0;

    tt1 = 2.0 / (ROCRAND_PI_DOUBLE * 0.147) + 0.5 * lnx;
    tt2 = 1.0 / (0.147) * lnx;

    return(sgn * sqrt(-tt1 + sqrt(tt1 * tt1 - tt2)));
}

// TODO: Improve implementation
/*
    float2 result;
    while (true)
    {
        float u = uniform(g);
        float v = uniform(g);
        float s = u * u + v * v;
        if (s < 1)
        {
            float multiplier = sqrtf(-2.0f * logf(s) / s);
            result.x = u * s;
            result.y = v * s;
            return result;
        }
    }
*/
__host__ __device__ float2 marsaglia(unsigned int x, unsigned int y)
{
    float2 result;
    float u = nextafterf(x * ROCRAND_2POW32_INV, 1.0f) * 2.0f - 1.0f;
    float v = nextafterf(y * ROCRAND_2POW32_INV, 1.0f) * 2.0f - 1.0f;
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
    double u = nextafter(zx * ROCRAND_2POW53_INV_DOUBLE, 1.0) * 2.0 - 1.0;
    unsigned long long zy = (unsigned long long)xy.z ^
        ((unsigned long long)xy.w << (53 - 32));
    double v = nextafter(zy * ROCRAND_2POW53_INV_DOUBLE, 1.0) * 2.0 - 1.0;
    double s = u * u + v * v;
    double multiplier = sqrt(-2.0f * log(s) / s);
    result.x = u * s;
    result.y = v * s;
    return result;
}

#endif // ROCRAND_RNG_DISTRIBUTION_NORMAL_COMMON_H_
