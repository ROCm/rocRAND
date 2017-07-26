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

#ifndef ROCRAND_LOG_NORMAL_H_
#define ROCRAND_LOG_NORMAL_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include "rocrand_philox4x32_10.h"
#include "rocrand_mrg32k3a.h"
#include "rocrand_xorwow.h"

#include "rocrand_normal.h"

#ifndef ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE
FQUALIFIERS
float rocrand_log_normal(rocrand_state_philox4x32_10 * state, float mean, float stddev)
{
    typedef rocrand_device::detail::philox4x32_10_engine_boxmuller_helper bm_helper;

    if(bm_helper::is_float(state))
    {
        return expf(mean + (stddev * bm_helper::get_float(state)));
    }
    float2 r = rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
    bm_helper::save_float(state, r.y);
    return expf(mean + (stddev * r.x));
}
#endif // ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE

FQUALIFIERS
float2 rocrand_log_normal2(rocrand_state_philox4x32_10 * state, float mean, float stddev)
{
    float2 r = rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
    return float2 {
        expf(mean + (stddev * r.x)),
        expf(mean + (stddev * r.y))
    };
}

FQUALIFIERS
float4 rocrand_log_normal4(rocrand_state_philox4x32_10 * state, float mean, float stddev)
{
    float4 r = rocrand_device::detail::normal_distribution4(rocrand4(state));
    return float4 {
        expf(mean + (stddev * r.x)),
        expf(mean + (stddev * r.y)),
        expf(mean + (stddev * r.z)),
        expf(mean + (stddev * r.w))
    };
}

#ifndef ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE
FQUALIFIERS
double rocrand_log_normal_double(rocrand_state_philox4x32_10 * state, double mean, double stddev)
{
    typedef rocrand_device::detail::philox4x32_10_engine_boxmuller_helper bm_helper;

    if(bm_helper::is_double(state))
    {
        return exp(mean + (stddev * bm_helper::get_double(state)));
    }
    double2 r = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    bm_helper::save_double(state, r.y);
    return exp(mean + r.x * stddev);
}
#endif // ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE

FQUALIFIERS
double2 rocrand_log_normal_double2(rocrand_state_philox4x32_10 * state, double mean, double stddev)
{
    double2 r = rocrand_device::detail::normal_distribution_double2(rocrand4(state));
    return double2 {
        exp(mean + (stddev * r.x)),
        exp(mean + (stddev * r.y))
    };
}

FQUALIFIERS
double4 rocrand_log_normal_double4(rocrand_state_philox4x32_10 * state, double mean, double stddev)
{
    double2 r1, r2;
    r1 = rocrand_log_normal_double2(state, mean, stddev);
    r2 = rocrand_log_normal_double2(state, mean, stddev);
    return double4 {
        r1.x, r1.y, r2.x, r2.y
    };
}

#ifndef ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE
FQUALIFIERS
float rocrand_log_normal(rocrand_state_mrg32k3a * state, float mean, float stddev)
{
    typedef rocrand_device::detail::mrg32k3a_engine_boxmuller_helper bm_helper;

    if(bm_helper::is_float(state))
    {
        return expf(mean + (stddev * bm_helper::get_float(state)));
    }
    float2 r = rocrand_device::detail::mrg_normal_distribution2(rocrand(state), rocrand(state));
    bm_helper::save_float(state, r.y);
    return expf(mean + (stddev * r.x));
}
#endif // ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE

FQUALIFIERS
float2 rocrand_log_normal2(rocrand_state_mrg32k3a * state, float mean, float stddev)
{
    float2 r = rocrand_device::detail::mrg_normal_distribution2(rocrand(state), rocrand(state));
    return float2 {
        expf(mean + (stddev * r.x)),
        expf(mean + (stddev * r.y))
    };
}

#ifndef ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE
FQUALIFIERS
double rocrand_log_normal_double(rocrand_state_mrg32k3a * state, double mean, double stddev)
{
    typedef rocrand_device::detail::mrg32k3a_engine_boxmuller_helper bm_helper;

    if(bm_helper::is_double(state))
    {
        return exp(mean + (stddev * bm_helper::get_double(state)));
    }
    double2 r = rocrand_device::detail::mrg_normal_distribution_double2(rocrand(state), rocrand(state));
    bm_helper::save_double(state, r.y);
    return exp(mean + r.x * stddev);
}
#endif // ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE

FQUALIFIERS
double2 rocrand_log_normal_double2(rocrand_state_mrg32k3a * state, double mean, double stddev)
{
    double2 r = rocrand_device::detail::mrg_normal_distribution_double2(rocrand(state), rocrand(state));
    return double2 {
        exp(mean + (stddev * r.x)),
        exp(mean + (stddev * r.y))
    };
}

#ifndef ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE
FQUALIFIERS
float rocrand_log_normal(rocrand_state_xorwow * state, float mean, float stddev)
{
    typedef rocrand_device::detail::xorwow_engine_boxmuller_helper bm_helper;

    if(bm_helper::is_float(state))
    {
        return expf(mean + (stddev * bm_helper::get_float(state)));
    }
    float2 r = rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
    bm_helper::save_float(state, r.y);
    return expf(mean + (stddev * r.x));
}
#endif // ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE

FQUALIFIERS
float2 rocrand_log_normal2(rocrand_state_xorwow * state, float mean, float stddev)
{
    float2 r = rocrand_device::detail::normal_distribution2(rocrand(state), rocrand(state));
    return float2 {
        expf(mean + (stddev * r.x)),
        expf(mean + (stddev * r.y))
    };
}

#ifndef ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE
FQUALIFIERS
double rocrand_log_normal_double(rocrand_state_xorwow * state, double mean, double stddev)
{
    typedef rocrand_device::detail::xorwow_engine_boxmuller_helper bm_helper;

    if(bm_helper::is_double(state))
    {
        return exp(mean + (stddev * bm_helper::get_double(state)));
    }
    double2 r = rocrand_device::detail::normal_distribution_double2(
        uint4 { rocrand(state), rocrand(state), rocrand(state), rocrand(state) }
    );
    bm_helper::save_double(state, r.y);
    return exp(mean + (stddev * r.x));
}
#endif // ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE

FQUALIFIERS
double2 rocrand_log_normal_double2(rocrand_state_xorwow * state, double mean, double stddev)
{
    double2 r = rocrand_device::detail::normal_distribution_double2(
        uint4 { rocrand(state), rocrand(state), rocrand(state), rocrand(state) }
    );
    return double2 {
        exp(mean + (stddev * r.x)),
        exp(mean + (stddev * r.y))
    };
}

#endif // ROCRAND_LOG_NORMAL_H_
