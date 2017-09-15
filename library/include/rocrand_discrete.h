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

#ifndef ROCRAND_DISCRETE_H_
#define ROCRAND_DISCRETE_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include <math.h>

#include "rocrand_philox4x32_10.h"
#include "rocrand_mrg32k3a.h"
#include "rocrand_xorwow.h"
#include "rocrand_sobol32.h"
#include "rocrand_mtgp32.h"

#include "rocrand_uniform.h"
#include "rocrand_normal.h"
#include "rocrand_discrete_types.h"

// Alias method
//
// Walker, A. J.
// An Efficient Method for Generating Discrete Random Variables with General Distributions, 1977
//
// Vose M. D.
// A Linear Algorithm For Generating Random Numbers With a Given Distribution, 1991

namespace rocrand_device {
namespace detail {

FQUALIFIERS
unsigned int discrete_alias(const double x, const rocrand_discrete_distribution_st& dis)
{
    // Calculate value using Alias table

    // x is [0, 1)
    const double nx = dis.size * x;
    const double fnx = floor(nx);
    const double y = nx - fnx;
    const unsigned int i = static_cast<unsigned int>(fnx);
    return dis.offset + (y < dis.probability[i] ? i : dis.alias[i]);
}

FQUALIFIERS
unsigned int discrete_alias(const unsigned int r, const rocrand_discrete_distribution_st& dis)
{
    const double x = r * ROCRAND_2POW32_INV_DOUBLE;
    return discrete_alias(x, dis);
}

FQUALIFIERS
unsigned int discrete_cdf(const double x, const rocrand_discrete_distribution_st& dis)
{
    // Calculate value using binary search in CDF

    unsigned int min = 0;
    unsigned int max = dis.size - 1;
    do
    {
        const unsigned int center = (min + max) / 2;
        const double p = dis.cdf[center];
        if (x > p)
        {
            min = center + 1;
        }
        else
        {
            max = center;
        }
    }
    while (min != max);

    return dis.offset + min;
}

FQUALIFIERS
unsigned int discrete_cdf(const unsigned int r, const rocrand_discrete_distribution_st& dis)
{
    const double x = r * ROCRAND_2POW32_INV_DOUBLE;
    return discrete_cdf(x, dis);
}

} // end namespace detail
} // end namespace rocrand_device

/** \rocrand_internal \addtogroup rocranddevice
 *
 *  @{
 */

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using Philox generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state - Pointer to a state to use
 * \param discrete_distribution - Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
FQUALIFIERS
unsigned int rocrand_discrete(rocrand_state_philox4x32_10 * state, const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns four discrete distributed <tt>unsigned int</tt> values.
 *
 * Returns four <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using Philox generator in \p state, and increments
 * the position of the generator by four.
 *
 * \param state - Pointer to a state to use
 * \param discrete_distribution - Related discrete distribution
 *
 * \return Four <tt>unsigned int</tt> values distributed according to \p discrete_distribution as \p uint4
 */
FQUALIFIERS
uint4 rocrand_discrete4(rocrand_state_philox4x32_10 * state, const rocrand_discrete_distribution discrete_distribution)
{
    const uint4 u4 = rocrand4(state);
    return uint4 {
        rocrand_device::detail::discrete_alias(u4.x, *discrete_distribution),
        rocrand_device::detail::discrete_alias(u4.y, *discrete_distribution),
        rocrand_device::detail::discrete_alias(u4.z, *discrete_distribution),
        rocrand_device::detail::discrete_alias(u4.w, *discrete_distribution)
    };
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using MRG32k3a generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state - Pointer to a state to use
 * \param discrete_distribution - Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
FQUALIFIERS
unsigned int rocrand_discrete(rocrand_state_mrg32k3a * state, const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using XORWOW generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state - Pointer to a state to use
 * \param discrete_distribution - Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
FQUALIFIERS
unsigned int rocrand_discrete(rocrand_state_xorwow * state, const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_alias(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using MTGP32 generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state - Pointer to a state to use
 * \param discrete_distribution - Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
FQUALIFIERS
unsigned int rocrand_discrete(rocrand_state_mtgp32 * state, const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
}

/**
 * \brief Returns a discrete distributed <tt>unsigned int</tt> value.
 *
 * Returns a <tt>unsigned int</tt> distributed according to with discrete distribution
 * \p discrete_distribution using SOBOL32 generator in \p state, and increments
 * the position of the generator by one.
 *
 * \param state - Pointer to a state to use
 * \param discrete_distribution - Related discrete distribution
 *
 * \return <tt>unsigned int</tt> value distributed according to \p discrete_distribution
 */
FQUALIFIERS
unsigned int rocrand_discrete(rocrand_state_sobol32 * state, const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete_cdf(rocrand(state), *discrete_distribution);
}

#endif // ROCRAND_DISCRETE_H_

/** @} */ // end of group rocranddevice
