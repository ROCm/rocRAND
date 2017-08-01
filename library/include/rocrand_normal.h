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

/** @addtogroup device
 *  
 *  @{
 */

#include "rocrand_philox4x32_10.h"
#include "rocrand_mrg32k3a.h"
#include "rocrand_xorwow.h"

#include "rocrand_uniform.h"

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
float2 mrg_box_muller(float x, float y)
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

FQUALIFIERS
double2 mrg_box_muller_double(double x, double y)
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
    
FQUALIFIERS
float normal_distribution(unsigned int x)
{
    float s = -ROCRAND_SQRT2;
    unsigned int z = x;
    if(z > 0x80000000) {
        z = 0xffffffff - z;
        s = -s;
    }
    float p = ::rocrand_device::detail::uniform_distribution(z);
    float v = s * erfcinvf(2.0f * p);
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
    double s = -ROCRAND_SQRT2_DOUBLE;
    unsigned int z = x;
    if(z > 0x80000000) {
        z = 0xffffffff - z;
        s = -s;
    }
    double p = ::rocrand_device::detail::uniform_distribution_double(z);
    double v = s * erfcinv(2.0 * p);
    return v;
}



FQUALIFIERS
double2 normal_distribution_double2(uint4 v)
{
    return ::rocrand_device::detail::box_muller_double(v);
}

FQUALIFIERS
float2 mrg_normal_distribution2(unsigned long long v1, unsigned long long v2)
{
    float x = rocrand_device::detail::mrg_uniform_distribution(v1);
    float y = rocrand_device::detail::mrg_uniform_distribution(v2);
    return ::rocrand_device::detail::mrg_box_muller(x, y);
}

FQUALIFIERS
double2 mrg_normal_distribution_double2(unsigned long long v1, unsigned long long v2)
{
    double x = rocrand_device::detail::mrg_uniform_distribution(v1);
    double y = rocrand_device::detail::mrg_uniform_distribution(v2);
    return ::rocrand_device::detail::mrg_box_muller_double(x, y);
}

} // end namespace detail
} // end namespace rocrand_device

/**
 * \brief Return a normally distributed float from a Philox Generator.
 *
 * Return a normally distributed float with mean \p 0.0f and 
 * standard deviation \p 1.0f from \p state, and increments 
 * position of generator by one. Box-Muller implementation
 * is used to generate normally distributed floats.
 *
 * \param state - Pointer to state to update
 *
 * \return normally distributed float with mean \p 0.0f and 
 * standard deviation \p 1.0f
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
 * \brief Return two normally distributed floats from a Philox Generator.
 *
 * Return two normally distributed floats with mean \p 0.0f and 
 * standard deviation \p 1.0f from \p state, and increments 
 * position of generator by two. Box-Muller implementation
 * is used to generate normally distributed floats.
 *
 * \param state - Pointer to state to update
 *
 * \return normally two distributed floats with mean \p 0.0f and 
 * standard deviation \p 1.0f
 */
FQUALIFIERS
float2 rocrand_normal2(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
}

/**
 * \brief Return four normally distributed floats from a Philox Generator.
 *
 * Return four normally distributed floats with mean \p 0.0f and 
 * standard deviation \p 1.0f from \p state, and increments 
 * position of generator by four. Box-Muller implementation
 * is used to generate normally distributed floats.
 *
 * \param state - Pointer to state to update
 *
 * \return normally four distributed floats with mean \p 0.0f and 
 * standard deviation \p 1.0f
 */
FQUALIFIERS
float4 rocrand_normal4(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::normal_distribution4(rocrand4(state));
}

/**
 * \brief Return a normally distributed double from a Philox Generator.
 *
 * Return a normally distributed double with mean \p 0.0 and 
 * standard deviation \p 1.0 from \p state, and increments 
 * position of generator by one. Box-Muller implementation
 * is used to generate normally distributed doubles.
 *
 * \param state - Pointer to state to update
 *
 * \return normally distributed double with mean \p 0.0 and 
 * standard deviation \p 1.0
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
 * \brief Return two normally distributed doubles from a Philox Generator.
 *
 * Return two normally distributed doubles with mean \p 0.0 and 
 * standard deviation \p 1.0 from \p state, and increments 
 * position of generator by two. Box-Muller implementation
 * is used to generate normally distributed doubles.
 *
 * \param state - Pointer to state to update
 *
 * \return normally two distributed doubles with mean \p 0.0 and 
 * standard deviation \p 1.0
 */
FQUALIFIERS
double2 rocrand_normal_double2(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::normal_distribution_double2(rocrand4(state));
}

/**
 * \brief Return four normally distributed doubles from a Philox Generator.
 *
 * Return four normally distributed doubles with mean \p 0.0 and 
 * standard deviation \p 1.0 from \p state, and increments 
 * position of generator by four. Box-Muller implementation
 * is used to generate normally distributed doubles.
 *
 * \param state - Pointer to state to update
 *
 * \return normally four distributed doubles with mean \p 0.0 and 
 * standard deviation \p 1.0
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
 * \brief Return a normally distributed float from a MRG32K3A Generator.
 *
 * Return a normally distributed float with mean \p 0.0f and 
 * standard deviation \p 1.0f from \p state, and increments 
 * position of generator by one. Box-Muller implementation
 * is used to generate normally distributed floats.
 *
 * \param state - Pointer to state to update
 *
 * \return normally distributed float with mean \p 0.0f and 
 * standard deviation \p 1.0f
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
 * \brief Return two normally distributed floats from a MRG32K3A Generator.
 *
 * Return two normally distributed floats with mean \p 0.0f and 
 * standard deviation \p 1.0f from \p state, and increments 
 * position of generator by two. Box-Muller implementation
 * is used to generate normally distributed floats.
 *
 * \param state - Pointer to state to update
 *
 * \return normally two distributed floats with mean \p 0.0f and 
 * standard deviation \p 1.0f
 */
FQUALIFIERS
float2 rocrand_normal2(rocrand_state_mrg32k3a * state)
{
    return rocrand_device::detail::mrg_normal_distribution2(rocrand(state), rocrand(state));
}

/**
 * \brief Return a normally distributed double from a MRG32K3A Generator.
 *
 * Return a normally distributed double with mean \p 0.0 and 
 * standard deviation \p 1.0 from \p state, and increments 
 * position of generator by one. Box-Muller implementation
 * is used to generate normally distributed doubles.
 *
 * \param state - Pointer to state to update
 *
 * \return normally distributed double with mean \p 0.0 and 
 * standard deviation \p 1.0
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
 * \brief Return two normally distributed doubles from a MRG32K3A Generator.
 *
 * Return two normally distributed doubles with mean \p 0.0 and 
 * standard deviation \p 1.0 from \p state, and increments 
 * position of generator by two. Box-Muller implementation
 * is used to generate normally distributed doubles.
 *
 * \param state - Pointer to state to update
 *
 * \return normally two distributed doubles with mean \p 0.0 and 
 * standard deviation \p 1.0
 */
FQUALIFIERS
double2 rocrand_normal_double2(rocrand_state_mrg32k3a * state)
{
    return rocrand_device::detail::mrg_normal_distribution_double2(rocrand(state), rocrand(state));
}

/**
 * \brief Return a normally distributed float from a XORWOW Generator.
 *
 * Return a normally distributed float with mean \p 0.0f and 
 * standard deviation \p 1.0f from \p state, and increments 
 * position of generator by one. Box-Muller implementation
 * is used to generate normally distributed floats.
 *
 * \param state - Pointer to state to update
 *
 * \return normally distributed float with mean \p 0.0f and 
 * standard deviation \p 1.0f
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
 * \brief Return two normally distributed floats from a XORWOW Generator.
 *
 * Return two normally distributed floats with mean \p 0.0f and 
 * standard deviation \p 1.0f from \p state, and increments 
 * position of generator by two. Box-Muller implementation
 * is used to generate normally distributed floats.
 *
 * \param state - Pointer to state to update
 *
 * \return normally two distributed floats with mean \p 0.0f and 
 * standard deviation \p 1.0f
 */
FQUALIFIERS
float2 rocrand_normal2(rocrand_state_xorwow * state)
{
    return rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
}

/**
 * \brief Return a normally distributed double from a XORWOW Generator.
 *
 * Return a normally distributed double with mean \p 0.0 and 
 * standard deviation \p 1.0 from \p state, and increments 
 * position of generator by one. Box-Muller implementation
 * is used to generate normally distributed doubles.
 *
 * \param state - Pointer to state to update
 *
 * \return normally distributed double with mean \p 0.0 and 
 * standard deviation \p 1.0
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
 * \brief Return two normally distributed doubles from a XORWOW Generator.
 *
 * Return two normally distributed doubles with mean \p 0.0 and 
 * standard deviation \p 1.0 from \p state, and increments 
 * position of generator by two. Box-Muller implementation
 * is used to generate normally distributed doubles.
 *
 * \param state - Pointer to state to update
 *
 * \return normally two distributed doubles with mean \p 0.0 and 
 * standard deviation \p 1.0
 */
FQUALIFIERS
double2 rocrand_normal_double2(rocrand_state_xorwow * state)
{
    return rocrand_device::detail::normal_distribution_double2(
        uint4 { rocrand(state), rocrand(state), rocrand(state), rocrand(state) }
    );
}

/**
 * \brief Return a normally distributed float from a SOBOL32 Generator.
 *
 * Return a normally distributed float with mean \p 0.0f and 
 * standard deviation \p 1.0f from \p state, and increments 
 * position of generator by one. 
 *
 * \param state - Pointer to state to update
 *
 * \return normally distributed float with mean \p 0.0f and 
 * standard deviation \p 1.0f
 */
FQUALIFIERS
float rocrand_normal(rocrand_state_sobol32 * state)
{
    return rocrand_device::detail::normal_distribution(rocrand(state));
}

/**
 * \brief Return a normally distributed double from a SOBOL32 Generator.
 *
 * Return a normally distributed double with mean \p 0.0 and 
 * standard deviation \p 1.0 from \p state, and increments 
 * position of generator by one. 
 *
 * \param state - Pointer to state to update
 *
 * \return normally distributed double with mean \p 0.0 and 
 * standard deviation \p 1.0
 */
FQUALIFIERS
double rocrand_normal_double(rocrand_state_sobol32 * state)
{
    return rocrand_device::detail::normal_distribution_double(rocrand(state));
}

#endif // ROCRAND_NORMAL_H_

/** @} */ // end of group device
