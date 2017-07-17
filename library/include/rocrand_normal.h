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

#ifndef ROCRAND_NORMAL_H_
#define ROCRAND_NORMAL_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include "rocrand_philox4x32_10.h"

namespace rocrand_device {
namespace detail {

FQUALIFIERS
float2 box_muller(unsigned int x, unsigned int y)
{
    float2 result;
    float u = nextafterf(x * ROCRAND_2POW32_INV, 1.0f);
    float v = nextafterf(y * ROCRAND_2POW32_INV_2PI, ROCRAND_2PI);
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

FQUALIFIERS
double2 box_muller_double(uint4 v)
{
    double2 result;
    unsigned long long v1 = (unsigned long long)v.x ^
        ((unsigned long long)v.y << (53 - 32));
    double u = nextafter(v1 * ROCRAND_2POW53_INV_DOUBLE, 1.0);
    unsigned long long v2 = (unsigned long long)v.z ^
        ((unsigned long long)v.w << (53 - 32));
    double w = nextafter(v2 * (ROCRAND_2POW53_INV_DOUBLE * 2.0), 2.0);
    double s = sqrt(-2.0 * log(u));
    #ifdef __HIP_DEVICE_COMPILE__
        sincospi(w, &result.x, &result.y);
        result.x *= s;
        result.y *= s;
    #else
        result.x = sin(w * ROCRAND_PI_DOUBLE) * s;
        result.y = cos(w * ROCRAND_PI_DOUBLE) * s;
    #endif
    return result;
}

FQUALIFIERS
float2 normal_distribution2(unsigned int v1, unsigned int v2)
{
    return ::rocrand_device::detail::box_muller(v1, v2);
}

FQUALIFIERS
float4 normal_distribution4(uint4 v)
{
    float2 r1 = ::rocrand_device::detail::box_muller(v.x, v.y);
    float2 r2 = ::rocrand_device::detail::box_muller(v.z, v.w);
    return float4{
        r1.x,
        r1.y,
        r2.x,
        r2.y
    };
}

FQUALIFIERS
double2 normal_distribution_double2(uint4 v)
{
    return ::rocrand_device::detail::box_muller_double(v);
}

} // end namespace detail
} // end namespace rocrand_device

FQUALIFIERS
float rocrand_normal(rocrand_state_philox4x32_10 * state)
{
    typedef rocrand_device::detail::philox4x32_10_engine_boxmuller_helper bm_helper;

    if(bm_helper::is_float(state))
    {
        return bm_helper::get_float(state);
    }
    float2 r = rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
    bm_helper::save_float(state, r.y);
    return r.x;
}

FQUALIFIERS
float2 rocrand_normal2(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
}

FQUALIFIERS
float4 rocrand_normal4(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::normal_distribution4(rocrand4(state));
}

FQUALIFIERS
double rocrand_normal_double(rocrand_state_philox4x32_10 * state)
{
    typedef rocrand_device::detail::philox4x32_10_engine_boxmuller_helper bm_helper;

    if(bm_helper::is_double(state))
    {
        return bm_helper::get_double(state);
    }
    double2 r = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    bm_helper::save_double(state, r.y);
    return r.x;
}

FQUALIFIERS
double2 rocrand_normal_double2(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::normal_distribution_double2(rocrand4(state));
}

#endif // ROCRAND_NORMAL_H_
