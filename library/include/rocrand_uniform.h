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

#ifndef ROCRAND_UNIFORM_H_
#define ROCRAND_UNIFORM_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include "rocrand_philox4x32_10.h"
#include "rocrand_mrg32k3a.h"
#include "rocrand_xorwow.h"

namespace rocrand_device {
namespace detail {

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0f and 1.0f, excluding 0.0f and including 1.0f.
FQUALIFIERS
float uniform_distribution(unsigned int v)
{
    return nextafterf(v * ROCRAND_2POW32_INV, 1.0f);
}

FQUALIFIERS
float4 uniform_distribution4(uint4 v)
{
   return float4 {
       nextafterf(v.x * ROCRAND_2POW32_INV, 1.0f),
       nextafterf(v.y * ROCRAND_2POW32_INV, 1.0f),
       nextafterf(v.z * ROCRAND_2POW32_INV, 1.0f),
       nextafterf(v.w * ROCRAND_2POW32_INV, 1.0f)
   };
}

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0 and 1.0, excluding 0.0 and including 1.0.
FQUALIFIERS
double uniform_distribution_double(unsigned int v)
{
    return nextafter(v * static_cast<double>(ROCRAND_2POW32_INV), 1.0);
}

FQUALIFIERS
double uniform_distribution_double(unsigned long long v)
{
    return nextafter(
        // 2^53 is the biggest int that can be stored in double, such
        // that it and all smaller integers can be stored in double
        (v >> 11) * ROCRAND_2POW53_INV_DOUBLE, 1.0
    );
}

FQUALIFIERS
double4 uniform_distribution_double4(uint4 v)
{
   return double4 {
       nextafter(v.x * static_cast<double>(ROCRAND_2POW32_INV), 1.0),
       nextafter(v.y * static_cast<double>(ROCRAND_2POW32_INV), 1.0),
       nextafter(v.z * static_cast<double>(ROCRAND_2POW32_INV), 1.0),
       nextafter(v.w * static_cast<double>(ROCRAND_2POW32_INV), 1.0)
   };
}

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0f and 1.0f, excluding 0.0f and including 1.0f (MRG32K3A).
FQUALIFIERS
float mrg_uniform_distribution(unsigned long long v)
{
    double ret = static_cast<double>(v) * ROCRAND_NORM_DOUBLE;
    return static_cast<float>(ret);
}

FQUALIFIERS
double mrg_uniform_distribution_double(unsigned long long v)
{
    double ret = static_cast<double>(v) * ROCRAND_NORM_DOUBLE;
    return ret;
}

} // end namespace detail
} // end namespace rocrand_device

FQUALIFIERS
float rocrand_uniform(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

FQUALIFIERS
float2 rocrand_uniform2(rocrand_state_philox4x32_10 * state)
{
    return float2 {
        rocrand_device::detail::uniform_distribution(rocrand(state)),
        rocrand_device::detail::uniform_distribution(rocrand(state))
    };
}

FQUALIFIERS
float4 rocrand_uniform4(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::uniform_distribution4(rocrand4(state));
}

FQUALIFIERS
double rocrand_uniform_double(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

FQUALIFIERS
double2 rocrand_uniform_double2(rocrand_state_philox4x32_10 * state)
{
    return double2 {
        rocrand_device::detail::uniform_distribution_double(rocrand(state)),
        rocrand_device::detail::uniform_distribution_double(rocrand(state))
    };
}

FQUALIFIERS
double4 rocrand_uniform_double4(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::uniform_distribution_double4(rocrand4(state));
}

FQUALIFIERS
float rocrand_uniform(rocrand_state_mrg32k3a * state)
{
    return rocrand_device::detail::mrg_uniform_distribution(rocrand(state));
}

FQUALIFIERS
double rocrand_uniform_double(rocrand_state_mrg32k3a * state)
{
    return rocrand_device::detail::mrg_uniform_distribution_double(rocrand(state));
}

FQUALIFIERS
float rocrand_uniform(rocrand_state_xorwow * state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

FQUALIFIERS
double rocrand_uniform_double(rocrand_state_xorwow * state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

#endif // ROCRAND_UNIFORM_H_
