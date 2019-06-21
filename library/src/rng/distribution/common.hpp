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

#ifndef ROCRAND_RNG_DISTRIBUTION_COMMON_H_
#define ROCRAND_RNG_DISTRIBUTION_COMMON_H_

#include <hip/hip_fp16.h>

#include "../common.hpp"

#if __HIP_DEVICE_COMPILE__ && (defined(__HIP_PLATFORM_HCC__) || (defined(__HIP_PLATFORM_NVCC__) && (__CUDA_ARCH__ >= 530)))
#define ROCRAND_HALF_MATH_SUPPORTED
#endif

FQUALIFIERS
__half uniform_distribution_half(unsigned short v)
{
    return __float2half(ROCRAND_2POW16_INV + (v * ROCRAND_2POW16_INV));
}

FQUALIFIERS
__half2 box_muller_half(unsigned short x, unsigned short y)
{
    #if defined(ROCRAND_HALF_MATH_SUPPORTED)
    __half u = __float2half(ROCRAND_2POW16_INV + (x * ROCRAND_2POW16_INV));
    __half v = __float2half(ROCRAND_2POW16_INV_2PI + (y * ROCRAND_2POW16_INV_2PI));
    __half s = hsqrt(__hmul(__float2half(-2.0f), hlog(u)));
    return __half2 {
        __hmul(hsin(v), s),
        __hmul(hcos(v), s)
    };
    #else
    float2 r;
    float u = ROCRAND_2POW16_INV + (x * ROCRAND_2POW16_INV);
    float v = ROCRAND_2POW16_INV_2PI + (y * ROCRAND_2POW16_INV_2PI);
    float s = sqrtf(-2.0f * logf(u));
    #ifdef __HIP_DEVICE_COMPILE__
        __sincosf(v, &r.x, &r.y);
        r.x *= s;
        r.y *= s;
    #else
        r.x = sinf(v) * s;
        r.y = cosf(v) * s;
    #endif
    return __half2 {
        __float2half(r.x),
        __float2half(r.y)
    };
    #endif
}

#endif // ROCRAND_RNG_DISTRIBUTION_COMMON_H_
