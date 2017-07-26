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

#ifndef ROCRAND_POISSON_H_
#define ROCRAND_POISSON_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include "rocrand_philox4x32_10.h"
#include "rocrand_mrg32k3a.h"
#include "rocrand_xorwow.h"

#include "rocrand_uniform.h"
#include "rocrand_normal.h"

namespace rocrand_device {
namespace detail {

constexpr double lambda_threshold_small = 64.0;
constexpr double lambda_threshold_huge  = 4000.0;

template<class State>
FQUALIFIERS
unsigned int poisson_distribution_small(State& state, double lambda)
{
    // Knuth's method

    const double limit = exp(-lambda);
    unsigned int k = 0;
    double product = 1.0;

    do
    {
        k++;
        product *= rocrand_uniform_double(state);
    }
    while (product > limit);

    return k - 1;
}

FQUALIFIERS
double log_factorial(const double n)
{
    return (n <= 1.0 ? 0.0 : lgamma(n + 1.0));
}

template<class State>
FQUALIFIERS
unsigned int poisson_distribution_large(State& state, double lambda)
{
    // Rejection method PA, A. C. Atkinson

    const double c = 0.767 - 3.36 / lambda;
    const double beta = ROCRAND_PI_DOUBLE / sqrt(3.0 * lambda);
    const double alpha = beta * lambda;
    const double k = log(c) - lambda - log(beta);
    const double log_lambda = log(lambda);

    while (true)
    {
        const double u = rocrand_uniform_double(state);
        const double x = (alpha - log((1.0 - u) / u)) / beta;
        const double n = floor(x + 0.5);
        if (n < 0)
        {
            continue;
        }
        const double v = rocrand_uniform_double(state);
        const double y = alpha - beta * x;
        const double t = 1.0 + exp(y);
        const double lhs = y + log(v / (t * t));
        const double rhs = k + n * log_lambda - log_factorial(n);
        if (lhs <= rhs)
        {
            return static_cast<unsigned int>(n);
        }
    }
}

template<class State>
FQUALIFIERS
unsigned int poisson_distribution_huge(State& state, double lambda)
{
    // Approximate Poisson distribution with normal distribution

    const double n = rocrand_normal_double(state);
    return static_cast<unsigned int>(round(sqrt(lambda) * n + lambda));
}

template<class State>
FQUALIFIERS
unsigned int poisson_distribution(State& state, double lambda)
{
    if (lambda < lambda_threshold_small)
    {
        return poisson_distribution_small(state, lambda);
    }
    else if (lambda <= lambda_threshold_huge)
    {
        return poisson_distribution_large(state, lambda);
    }
    else
    {
        return poisson_distribution_huge(state, lambda);
    }
}

} // end namespace detail
} // end namespace rocrand_device

#ifndef ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE
FQUALIFIERS
unsigned int rocrand_poisson(rocrand_state_philox4x32_10 * state, double lambda)
{
    return rocrand_device::detail::poisson_distribution(state, lambda);
}

FQUALIFIERS
uint4 rocrand_poisson4(rocrand_state_philox4x32_10 * state, double lambda)
{
    return uint4 {
        rocrand_device::detail::poisson_distribution(state, lambda),
        rocrand_device::detail::poisson_distribution(state, lambda),
        rocrand_device::detail::poisson_distribution(state, lambda),
        rocrand_device::detail::poisson_distribution(state, lambda)
    };
}
#endif // ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE

#ifndef ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE
FQUALIFIERS
unsigned int rocrand_poisson(rocrand_state_mrg32k3a * state, double lambda)
{
    return rocrand_device::detail::poisson_distribution(state, lambda);
}
#endif // ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE

#ifndef ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE
FQUALIFIERS
unsigned int rocrand_poisson(rocrand_state_xorwow * state, double lambda)
{
    return rocrand_device::detail::poisson_distribution(state, lambda);
}
#endif // ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE

#endif // ROCRAND_POISSON_H_
