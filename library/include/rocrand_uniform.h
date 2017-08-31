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

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

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

struct two_uints
{
    unsigned int x;
    unsigned int y;
};

union two_uints_to_ulong
{
    two_uints uint2_value;
    unsigned long long ulong_value;
};

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0f and 1.0f, excluding 0.0f and including 1.0f.
FQUALIFIERS
float uniform_distribution(unsigned int v)
{
    return ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV);
}

FQUALIFIERS
float4 uniform_distribution4(uint4 v)
{
   return float4 {
       ROCRAND_2POW32_INV + (v.x * ROCRAND_2POW32_INV),
       ROCRAND_2POW32_INV + (v.y * ROCRAND_2POW32_INV),
       ROCRAND_2POW32_INV + (v.z * ROCRAND_2POW32_INV),
       ROCRAND_2POW32_INV + (v.w * ROCRAND_2POW32_INV)
   };
}

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0 and 1.0, excluding 0.0 and including 1.0.
FQUALIFIERS
double uniform_distribution_double(unsigned int v)
{
    return ROCRAND_2POW32_INV_DOUBLE + (v * ROCRAND_2POW32_INV_DOUBLE);
}

FQUALIFIERS
double uniform_distribution_double(unsigned int v1, unsigned int v2)
{
    two_uints_to_ulong v;
    v.uint2_value.x = v1;
    v.uint2_value.y = (v2 >> 11);
    return ROCRAND_2POW53_INV_DOUBLE + (v.ulong_value * ROCRAND_2POW53_INV_DOUBLE);
}

FQUALIFIERS
double uniform_distribution_double(unsigned long long v)
{
    return ROCRAND_2POW53_INV_DOUBLE + (
        // 2^53 is the biggest int that can be stored in double, such
        // that it and all smaller integers can be stored in double
        (v >> 11) * ROCRAND_2POW53_INV_DOUBLE
    );
}

FQUALIFIERS
double2 uniform_distribution_double2(uint4 v)
{
    return double2 {
        uniform_distribution_double(v.x, v.y),
        uniform_distribution_double(v.z, v.w)
    };
}

FQUALIFIERS
double4 uniform_distribution_double4(uint4 v)
{
   return double4 {
       ROCRAND_2POW32_INV_DOUBLE + (v.x * ROCRAND_2POW32_INV_DOUBLE),
       ROCRAND_2POW32_INV_DOUBLE + (v.y * ROCRAND_2POW32_INV_DOUBLE),
       ROCRAND_2POW32_INV_DOUBLE + (v.z * ROCRAND_2POW32_INV_DOUBLE),
       ROCRAND_2POW32_INV_DOUBLE + (v.w * ROCRAND_2POW32_INV_DOUBLE)
   };
}

FQUALIFIERS
double4 uniform_distribution_double4(uint4 v1, uint4 v2)
{
    return double4 {
        uniform_distribution_double(v1.x, v1.y),
        uniform_distribution_double(v1.z, v1.w),
        uniform_distribution_double(v2.x, v2.y),
        uniform_distribution_double(v2.z, v2.w)
    };
}

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0f and 1.0f, excluding 0.0f and including 1.0f (MRG32K3A).
FQUALIFIERS
float mrg_uniform_distribution(unsigned int v)
{
    double ret = static_cast<double>(v) * ROCRAND_MRG32K3A_NORM_DOUBLE;
    return static_cast<float>(ret);
}

FQUALIFIERS
double mrg_uniform_distribution_double(unsigned int v)
{
    double ret = static_cast<double>(v) * ROCRAND_MRG32K3A_NORM_DOUBLE;
    return ret;
}

} // end namespace detail
} // end namespace rocrand_device

/**
 * \brief Return a uniformly distributed float from a Philox Generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from \p state, and increments position of generator by one.
 * Output range excludes \p 0.0f but includes \p 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
FQUALIFIERS
float rocrand_uniform(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Return two uniformly distributed floats from a Philox Generator.
 *
 * Return two uniformly distributed floats between \p 0.0f and \p 1.0f
 * from \p state, and increments position of generator by two.
 * Output range excludes \p 0.0f but includes \p 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return two uniformly distributed float between \p 0.0f and \p 1.0f
 */
FQUALIFIERS
float2 rocrand_uniform2(rocrand_state_philox4x32_10 * state)
{
    return float2 {
        rocrand_device::detail::uniform_distribution(rocrand(state)),
        rocrand_device::detail::uniform_distribution(rocrand(state))
    };
}

/**
 * \brief Return four uniformly distributed floats from a Philox Generator.
 *
 * Return four uniformly distributed floats between \p 0.0f and \p 1.0f
 * from \p state, and increments position of generator by four.
 * Output range excludes \p 0.0f but includes \p 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return four uniformly distributed float between \p 0.0f and \p 1.0f
 */
FQUALIFIERS
float4 rocrand_uniform4(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::uniform_distribution4(rocrand4(state));
}

/**
 * \brief Return a uniformly distributed double from a Philox Generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from \p state, and increments position of generator by one.
 * Output range excludes \p 0.0 but includes \p 1.0.
 *
 * \param state - Pointer to a state to use
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
FQUALIFIERS
double rocrand_uniform_double(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state), rocrand(state));
}

/**
 * \brief Return two uniformly distributed doubles from a Philox Generator.
 *
 * Return two uniformly distributed doubles between \p 0.0 and \p 1.0
 * from \p state, and increments position of generator by two.
 * Output range excludes \p 0.0 but includes \p 1.0.
 *
 * \param state - Pointer to a state to use
 *
 * \return two uniformly distributed float between \p 0.0 and \p 1.0
 */
FQUALIFIERS
double2 rocrand_uniform_double2(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::uniform_distribution_double2(rocrand4(state));
}

/**
 * \brief Return four uniformly distributed doubles from a Philox Generator.
 *
 * Return four uniformly distributed doubles between \p 0.0 and \p 1.0
 * from \p state, and increments position of generator by four.
 * Output range excludes \p 0.0 but includes \p 1.0.
 *
 * \param state - Pointer to a state to use
 *
 * \return four uniformly distributed doubles between \p 0.0 and \p 1.0
 */
FQUALIFIERS
double4 rocrand_uniform_double4(rocrand_state_philox4x32_10 * state)
{
    return rocrand_device::detail::uniform_distribution_double4(rocrand4(state), rocrand4(state));
}

/**
 * \brief Return a uniformly distributed float from a MRG32K3A Generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from \p state, and increments position of generator by one.
 * Output range excludes \p 0.0f but includes \p 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
FQUALIFIERS
float rocrand_uniform(rocrand_state_mrg32k3a * state)
{
    return rocrand_device::detail::mrg_uniform_distribution(rocrand(state));
}

/**
 * \brief Return a uniformly distributed double from a MRG32K3A Generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from \p state, and increments position of generator by one.
 * Output range excludes \p 0.0 but includes \p 1.0.
 *
 * \param state - Pointer to a state to use
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
FQUALIFIERS
double rocrand_uniform_double(rocrand_state_mrg32k3a * state)
{
    return rocrand_device::detail::mrg_uniform_distribution_double(rocrand(state));
}

/**
 * \brief Return a uniformly distributed float from a XORWOW Generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from \p state, and increments position of generator by one.
 * Output range excludes \p 0.0f but includes \p 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
FQUALIFIERS
float rocrand_uniform(rocrand_state_xorwow * state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Return a uniformly distributed double from a XORWOW Generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from \p state, and increments position of generator by one.
 * Output range excludes \p 0.0 but includes \p 1.0.
 *
 * \param state - Pointer to a state to use
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
FQUALIFIERS
double rocrand_uniform_double(rocrand_state_xorwow * state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state), rocrand(state));
}

/**
 * \brief Return a uniformly distributed float from a MTGP32 Generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from \p state, and increments position of generator by one.
 * Output range excludes \p 0.0f but includes \p 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
FQUALIFIERS
float rocrand_uniform(rocrand_state_mtgp32 * state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Return a uniformly distributed double from a MTGP32 Generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from \p state, and increments position of generator by one.
 * Output range excludes \p 0.0 but includes \p 1.0.
 *
 * \param state - Pointer to a state to use
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
FQUALIFIERS
double rocrand_uniform_double(rocrand_state_mtgp32 * state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

/**
 * \brief Return a uniformly distributed float from a SOBOL32 Generator.
 *
 * Return a uniformly distributed float between \p 0.0f and \p 1.0f
 * from \p state, and increments position of generator by one.
 * Output range excludes \p 0.0f but includes \p 1.0f.
 *
 * \param state - Pointer to a state to use
 *
 * \return uniformly distributed float between \p 0.0f and \p 1.0f
 */
FQUALIFIERS
float rocrand_uniform(rocrand_state_sobol32 * state)
{
    return rocrand_device::detail::uniform_distribution(rocrand(state));
}

/**
 * \brief Return a uniformly distributed double from a SOBOL32 Generator.
 *
 * Return a uniformly distributed double between \p 0.0 and \p 1.0
 * from \p state, and increments position of generator by one.
 * Output range excludes \p 0.0 but includes \p 1.0.
 *
 * \param state - Pointer to a state to use
 *
 * \return uniformly distributed double between \p 0.0 and \p 1.0
 */
FQUALIFIERS
double rocrand_uniform_double(rocrand_state_sobol32 * state)
{
    return rocrand_device::detail::uniform_distribution_double(rocrand(state));
}

#endif // ROCRAND_UNIFORM_H_

/** @} */ // end of group rocranddevice
