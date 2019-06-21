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

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

#include <math.h>

#include "rocrand_philox4x32_10.h"
#include "rocrand_mrg32k3a.h"
#include "rocrand_xorwow.h"
#include "rocrand_sobol32.h"
#include "rocrand_mtgp32.h"

#include "rocrand_uniform.h"

namespace rocrand_device {
namespace detail {

FQUALIFIERS
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

FQUALIFIERS
double2 box_muller_double(uint4 v)
{
    double2 result;
    unsigned long long v1 = (unsigned long long)v.x ^
        ((unsigned long long)v.y << (53 - 32));
    double u = ROCRAND_2POW53_INV_DOUBLE + (v1 * ROCRAND_2POW53_INV_DOUBLE);
    unsigned long long v2 = (unsigned long long)v.z ^
        ((unsigned long long)v.w << (53 - 32));
    double w = (ROCRAND_2POW53_INV_DOUBLE * 2.0) +
        (v2 * (ROCRAND_2POW53_INV_DOUBLE * 2.0));
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

FQUALIFIERS
float2 mrg_box_muller(unsigned int x, unsigned int y)
{
    float2 result;
    float u = rocrand_device::detail::mrg_uniform_distribution(x);
    float v = rocrand_device::detail::mrg_uniform_distribution(y) * ROCRAND_2PI;
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
double2 mrg_box_muller_double(unsigned int x, unsigned int y)
{
    double2 result;
    double u = rocrand_device::detail::mrg_uniform_distribution(x);
    double v = rocrand_device::detail::mrg_uniform_distribution(y) * 2.0;
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

FQUALIFIERS
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

FQUALIFIERS
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

FQUALIFIERS
float normal_distribution(unsigned int x)
{
    float p = ::rocrand_device::detail::uniform_distribution(x);
    float v = ROCRAND_SQRT2 * ::rocrand_device::detail::roc_f_erfinv(2.0f * p - 1.0f);
    return v;
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
double normal_distribution_double(unsigned int x)
{
    double p = ::rocrand_device::detail::uniform_distribution_double(x);
    double v = ROCRAND_SQRT2 * ::rocrand_device::detail::roc_d_erfinv(2.0 * p - 1.0);
    return v;
}

FQUALIFIERS
double2 normal_distribution_double2(uint4 v)
{
    return ::rocrand_device::detail::box_muller_double(v);
}

FQUALIFIERS
__half2 normal_distribution_half2(unsigned int v)
{
    return ::rocrand_device::detail::box_muller_half(
        static_cast<unsigned short>(v),
        static_cast<unsigned short>(v >> 16)
    );
}

FQUALIFIERS
float2 mrg_normal_distribution2(unsigned int v1, unsigned int v2)
{
    return ::rocrand_device::detail::mrg_box_muller(v1, v2);
}

FQUALIFIERS
double2 mrg_normal_distribution_double2(unsigned int v1, unsigned int v2)
{
    return ::rocrand_device::detail::mrg_box_muller_double(v1, v2);
}

FQUALIFIERS
__half2 mrg_normal_distribution_half2(unsigned int v)
{
    v = rocrand_device::detail::mrg_uniform_distribution_uint(v);
    return ::rocrand_device::detail::box_muller_half(
        static_cast<unsigned short>(v),
        static_cast<unsigned short>(v >> 16)
    );
}

} // end namespace detail
} // end namespace rocrand_device

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using Philox
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state - Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE
FQUALIFIERS
float rocrand_normal(rocrand_state_philox4x32_10 * state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_philox4x32_10> bm_helper;

    if(bm_helper::has_float(state))
    {
        return bm_helper::get_float(state);
    }
    float2 r = rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
    bm_helper::save_float(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using Philox
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state - Pointer to a state to use
 *
 * \return Two normally distributed \p float value as \p float2
 */
FQUALIFIERS
float2 rocrand_normal2(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
}

/**
 * \brief Returns four normally distributed \p float values.
 *
 * Generates and returns four normally distributed \p float values using Philox
 * generator in \p state, and increments position of the generator by four.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate four normally
 * distributed values, and returns them.
 *
 * \param state - Pointer to a state to use
 *
 * \return Four normally distributed \p float value as \p float4
 */
FQUALIFIERS
float4 rocrand_normal4(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::normal_distribution4(rocrand4(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using Philox
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state - Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE
FQUALIFIERS
double rocrand_normal_double(rocrand_state_philox4x32_10 * state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_philox4x32_10> bm_helper;

    if(bm_helper::has_double(state))
    {
        return bm_helper::get_double(state);
    }
    double2 r = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    bm_helper::save_double(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using Philox
 * generator in \p state, and increments position of the generator by four.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state - Pointer to a state to use
 *
 * \return Two normally distributed \p double values as \p double2
 */
FQUALIFIERS
double2 rocrand_normal_double2(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::normal_distribution_double2(rocrand4(state));
}

/**
 * \brief Returns four normally distributed \p double values.
 *
 * Generates and returns four normally distributed \p double values using Philox
 * generator in \p state, and increments position of the generator by eight.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate four normally
 * distributed values, and returns them.
 *
 * \param state - Pointer to a state to use
 *
 * \return Four normally distributed \p double values as \p double4
 */
FQUALIFIERS
double4 rocrand_normal_double4(rocrand_state_philox4x32_10 * state)
{
    double2 r1, r2;
    r1 = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    r2 = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    return double4 {
        r1.x, r1.y, r2.x, r2.y
    };
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using MRG32k3a
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state - Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE
FQUALIFIERS
float rocrand_normal(rocrand_state_mrg32k3a * state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_mrg32k3a> bm_helper;

    if(bm_helper::has_float(state))
    {
        return bm_helper::get_float(state);
    }
    float2 r = rocrand_device::detail::mrg_normal_distribution2(rocrand(state), rocrand(state));
    bm_helper::save_float(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using MRG32k3a
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state - Pointer to a state to use
 *
 * \return Two normally distributed \p float value as \p float2
 */
FQUALIFIERS
float2 rocrand_normal2(rocrand_state_mrg32k3a * state)
{
    return rocrand_device::detail::mrg_normal_distribution2(rocrand(state), rocrand(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using MRG32k3a
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state - Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE
FQUALIFIERS
double rocrand_normal_double(rocrand_state_mrg32k3a * state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_mrg32k3a> bm_helper;

    if(bm_helper::has_double(state))
    {
        return bm_helper::get_double(state);
    }
    double2 r = rocrand_device::detail::mrg_normal_distribution_double2(rocrand(state), rocrand(state));
    bm_helper::save_double(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using MRG32k3a
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state - Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
FQUALIFIERS
double2 rocrand_normal_double2(rocrand_state_mrg32k3a * state)
{
    return rocrand_device::detail::mrg_normal_distribution_double2(rocrand(state), rocrand(state));
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using XORWOW
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state - Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
#ifndef ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE
FQUALIFIERS
float rocrand_normal(rocrand_state_xorwow * state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_xorwow> bm_helper;

    if(bm_helper::has_float(state))
    {
        return bm_helper::get_float(state);
    }
    float2 r = rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
    bm_helper::save_float(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p float values.
 *
 * Generates and returns two normally distributed \p float values using XORWOW
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state - Pointer to a state to use
 *
 * \return Two normally distributed \p float values as \p float2
 */
FQUALIFIERS
float2 rocrand_normal2(rocrand_state_xorwow * state)
{
    return rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using XORWOW
 * generator in \p state, and increments position of the generator by two.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally distributed
 * values, returns first of them, and saves the second to be returned on the next call.
 *
 * \param state - Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
#ifndef ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE
FQUALIFIERS
double rocrand_normal_double(rocrand_state_xorwow * state)
{
    typedef rocrand_device::detail::engine_boxmuller_helper<rocrand_state_xorwow> bm_helper;

    if(bm_helper::has_double(state))
    {
        return bm_helper::get_double(state);
    }
    double2 r = rocrand_device::detail::normal_distribution_double2(
        uint4 { rocrand(state), rocrand(state), rocrand(state), rocrand(state) }
    );
    bm_helper::save_double(state, r.y);
    return r.x;
}
#endif // ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE

/**
 * \brief Returns two normally distributed \p double values.
 *
 * Generates and returns two normally distributed \p double values using XORWOW
 * generator in \p state, and increments position of the generator by four.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 * The function uses the Box-Muller transform method to generate two normally
 * distributed values, and returns both of them.
 *
 * \param state - Pointer to a state to use
 *
 * \return Two normally distributed \p double value as \p double2
 */
FQUALIFIERS
double2 rocrand_normal_double2(rocrand_state_xorwow * state)
{
    return rocrand_device::detail::normal_distribution_double2(
        uint4 { rocrand(state), rocrand(state), rocrand(state), rocrand(state) }
    );
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using MTGP32
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
FQUALIFIERS
float rocrand_normal(rocrand_state_mtgp32 * state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using MRG32k3a
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
FQUALIFIERS
double rocrand_normal_double(rocrand_state_mtgp32 * state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p float value.
 *
 * Generates and returns a normally distributed \p float value using SOBOL32
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return Normally distributed \p float value
 */
FQUALIFIERS
float rocrand_normal(rocrand_state_sobol32 * state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Returns a normally distributed \p double value.
 *
 * Generates and returns a normally distributed \p double value using SOBOL32
 * generator in \p state, and increments position of the generator by one.
 * Used normal distribution has mean value equal to 0.0f, and standard deviation
 * equal to 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return Normally distributed \p double value
 */
FQUALIFIERS
double rocrand_normal_double(rocrand_state_sobol32 * state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

#endif // ROCRAND_NORMAL_H_

/** @} */ // end of group rocranddevice
