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
double lgamma_approx(const double x)
{
    // Lanczos approximation (g = 7, n = 9)

    const double z = x - 1.0;

    const int g = 7;
    const int n = 9;
    const double coefs[n] = {
        0.99999999999980993227684700473478,
        676.520368121885098567009190444019,
        -1259.13921672240287047156078755283,
        771.3234287776530788486528258894,
        -176.61502916214059906584551354,
        12.507343278686904814458936853,
        -0.13857109526572011689554707,
        9.984369578019570859563e-6,
        1.50563273514931155834e-7
    };
    double sum = 0.0;
    #pragma unroll
    for (int i = n - 1; i > 0; i--)
    {
        sum += coefs[i] / (z + i);
    }
    sum += coefs[0];

    const double log_sqrt_2_pi = 0.9189385332046727418;
    const double e = 2.718281828459045090796;
    return (log_sqrt_2_pi + log(sum) - g) + (z + 0.5) * log((z + g + 0.5) / e);
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
        const double rhs = k + n * log_lambda - lgamma_approx(n + 1.0);
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

template<class State>
FQUALIFIERS
unsigned int poisson_distribution_itr(State& state, double lambda)
{
    // Algorithm ITR
    // George S. Fishman
    // Discrete-Event Simulation: Modeling, Programming, and Analysis
    // p. 333
    double L;
    double x = 1.0;
    double y = 1.0;
    int k = 0;
    int pow = 0;
    // Algorithm ITR uses u from (0, 1) and uniform_double returns (0, 1]
    // Change u to ensure that 1 is never generated,
    // otherwise the inner loop never ends.
    double u = rocrand_uniform_double(state) - ROCRAND_2POW32_INV_DOUBLE / 2.0;
    double upow = pow + 500.0;
    double ex = exp(-500.0);
    do{
        if (lambda > upow)
            L = ex;
        else
            L = exp((double)(pow - lambda));

        x *= L;
        y *= L;
        pow += 500;
        while (u > y)
        {
            k++;
            x *= ((double)lambda / (double) k);
            y += x;
        }
    } while((double)pow < lambda);
    return k;
}

template<class State>
FQUALIFIERS
unsigned int poisson_distribution_inv(State& state, double lambda)
{
    if (lambda < 1000.0)
    {
        return poisson_distribution_itr(state, lambda);
    }
    else
    {
        return poisson_distribution_huge(state, lambda);
    }
}

} // end namespace detail
} // end namespace rocrand_device

/**
 * \brief Returns a Poisson-distributed <tt>unsigned int</tt> using Philox generator.
 *
 * Generates and returns Poisson-distributed distributed random <tt>unsigned int</tt>
 * values using Philox generator in \p state. State is incremented by a variable amount.
 *
 * \param state - Pointer to a state to use
 * \param lambda - Lambda parameter of the Poisson distribution
 *
 * \return Poisson-distributed <tt>unsigned int</tt>
 */
#ifndef ROCRAND_DETAIL_PHILOX_BM_NOT_IN_STATE
FQUALIFIERS
unsigned int rocrand_poisson(rocrand_state_philox4x32_10 * state, double lambda)
{
    return rocrand_device::detail::poisson_distribution(state, lambda);
}

/**
 * \brief Returns four Poisson-distributed <tt>unsigned int</tt> values using Philox generator.
 *
 * Generates and returns Poisson-distributed distributed random <tt>unsigned int</tt>
 * values using Philox generator in \p state. State is incremented by a variable amount.
 *
 * \param state - Pointer to a state to use
 * \param lambda - Lambda parameter of the Poisson distribution
 *
 * \return Four Poisson-distributed <tt>unsigned int</tt> values as \p uint4
 */
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

/**
 * \brief Returns a Poisson-distributed <tt>unsigned int</tt> using MRG32k3a generator.
 *
 * Generates and returns Poisson-distributed distributed random <tt>unsigned int</tt>
 * values using MRG32k3a generator in \p state. State is incremented by a variable amount.
 *
 * \param state - Pointer to a state to use
 * \param lambda - Lambda parameter of the Poisson distribution
 *
 * \return Poisson-distributed <tt>unsigned int</tt>
 */
#ifndef ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE
FQUALIFIERS
unsigned int rocrand_poisson(rocrand_state_mrg32k3a * state, double lambda)
{
    return rocrand_device::detail::poisson_distribution(state, lambda);
}
#endif // ROCRAND_DETAIL_MRG32K3A_BM_NOT_IN_STATE

/**
 * \brief Returns a Poisson-distributed <tt>unsigned int</tt> using XORWOW generator.
 *
 * Generates and returns Poisson-distributed distributed random <tt>unsigned int</tt>
 * values using XORWOW generator in \p state. State is incremented by a variable amount.
 *
 * \param state - Pointer to a state to use
 * \param lambda - Lambda parameter of the Poisson distribution
 *
 * \return Poisson-distributed <tt>unsigned int</tt>
 */
#ifndef ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE
FQUALIFIERS
unsigned int rocrand_poisson(rocrand_state_xorwow * state, double lambda)
{
    return rocrand_device::detail::poisson_distribution(state, lambda);
}
#endif // ROCRAND_DETAIL_XORWOW_BM_NOT_IN_STATE

/**
 * \brief Returns a Poisson-distributed <tt>unsigned int</tt> using MTGP32 generator.
 *
 * Generates and returns Poisson-distributed distributed random <tt>unsigned int</tt>
 * values using MTGP32 generator in \p state. State is incremented by one position.
 *
 * \param state - Pointer to a state to use
 * \param lambda - Lambda parameter of the Poisson distribution
 *
 * \return Poisson-distributed <tt>unsigned int</tt>
 */
FQUALIFIERS
unsigned int rocrand_poisson(rocrand_state_mtgp32 * state, double lambda)
{
    return rocrand_device::detail::poisson_distribution_inv(state, lambda);
}

/**
 * \brief Returns a Poisson-distributed <tt>unsigned int</tt> using SOBOL32 generator.
 *
 * Generates and returns Poisson-distributed distributed random <tt>unsigned int</tt>
 * values using SOBOL32 generator in \p state. State is incremented by one position.
 *
 * \param state - Pointer to a state to use
 * \param lambda - Lambda parameter of the Poisson distribution
 *
 * \return Poisson-distributed <tt>unsigned int</tt>
 */
FQUALIFIERS
unsigned int rocrand_poisson(rocrand_state_sobol32 * state, double lambda)
{
    return rocrand_device::detail::poisson_distribution_inv(state, lambda);
}

#endif // ROCRAND_POISSON_H_

/** @} */ // end of group rocranddevice
