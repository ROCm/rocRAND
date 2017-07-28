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

#include "rocrand_philox4x32_10.h"
#include "rocrand_mrg32k3a.h"
#include "rocrand_xorwow.h"

#include "rocrand_uniform.h"
#include "rocrand_normal.h"
#include "rocrand_discrete_types.h"

// George Marsaglia, Wai Wan Tsang, Jingbo Wang
// Fast Generation of Discrete Random Variables
// Journal of Statistical Software, 2004
// https://www.jstatsoft.org/article/view/v011i03

// Only square histogram from Method II is used here (without J[256]).
// Instead of increasing performance, J[256] makes the algorithm slower on GPUs.

namespace rocrand_device {
namespace detail {

template<class State>
FQUALIFIERS
unsigned int small_discrete(State& state, const rocrand_discrete_distribution_st& dis)
{
    // Calculate value using square histrogram

    const unsigned int r = rocrand(state);
    // [0, 1)
    const double u = r * 2.3283064365386963e-10;
    const unsigned int j = static_cast<unsigned int>(floor(dis.size * u));
    return dis.offset + (u < dis.V[j] ? j : dis.K[j]);
}

template<class State>
FQUALIFIERS
unsigned int large_discrete(State& state, const rocrand_discrete_distribution_st& dis)
{
    // Approximate Poisson distribution with normal distribution

    const double n = rocrand_normal_double(state);
    return static_cast<unsigned int>(round(dis.normal_stddev * n + dis.normal_mean));
}

template<class State>
FQUALIFIERS
unsigned int discrete(State& state, const rocrand_discrete_distribution_st& dis)
{
    if (dis.size != 0)
    {
        return small_discrete(state, dis);
    }
    else
    {
        return large_discrete(state, dis);
    }
}

} // end namespace detail
} // end namespace rocrand_device

#ifndef ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE
FQUALIFIERS
unsigned int rocrand_discrete(rocrand_state_philox4x32_10 * state, const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete(state, *discrete_distribution);
}

FQUALIFIERS
uint4 rocrand_discrete4(rocrand_state_philox4x32_10 * state, const rocrand_discrete_distribution discrete_distribution)
{
    // Naive implemetation
    return uint4 {
        rocrand_device::detail::discrete(state, *discrete_distribution),
        rocrand_device::detail::discrete(state, *discrete_distribution),
        rocrand_device::detail::discrete(state, *discrete_distribution),
        rocrand_device::detail::discrete(state, *discrete_distribution)
    };
}
#endif // ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE

#ifndef ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE
FQUALIFIERS
unsigned int rocrand_discrete(rocrand_state_mrg32k3a * state, const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete(state, *discrete_distribution);
}
#endif // ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE

#ifndef ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE
FQUALIFIERS
unsigned int rocrand_discrete(rocrand_state_xorwow * state, const rocrand_discrete_distribution discrete_distribution)
{
    return rocrand_device::detail::discrete(state, *discrete_distribution);
}
#endif // ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE

#endif // ROCRAND_DISCRETE_H_
