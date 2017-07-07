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

#ifndef ROCRAND_RNG_DISTRIBUTION_POISSON_H_
#define ROCRAND_RNG_DISTRIBUTION_POISSON_H_

#include <climits>

#include "common.hpp"
#include "normal.hpp"
#include "uniform.hpp"

template<class T>
class poisson_distribution;

template<>
class poisson_distribution<unsigned int>
{
public:

    __host__ __device__
    poisson_distribution(double lambda)
        : lambda(lambda) {}

    template<class Generator>
    __host__ __device__
    unsigned int operator()(Generator& g)
    {
        if (lambda < lambda_threshold_small)
        {
            return generate_small(g);
        }
        else if (lambda <= lambda_threshold_huge)
        {
            return generate_large(g);
        }
        else
        {
            return generate_huge(g);
        }
    }

private:

    static constexpr double lambda_threshold_small = 64.0;
    static constexpr double lambda_threshold_huge  = 4000.0;

    double lambda;

    template<class Generator>
    __host__ __device__
    unsigned int generate_small(Generator& g)
    {
        // Knuth's method

        const double limit = exp(-lambda);
        unsigned int k = 0;
        double product = 1.0;

        uniform_distribution<double> uniform;
        do
        {
            k++;
            product *= uniform(g());
        }
        while (product > limit);

        return k - 1;
    }

    template<class Generator>
    __host__ __device__
    unsigned int generate_large(Generator& g)
    {
        // Rejection method PA, A. C. Atkinson

        const double c = 0.767 - 3.36 / lambda;
        const double beta = ROC_PI_DOUBLE / sqrt(3.0 * lambda);
        const double alpha = beta * lambda;
        const double k = log(c) - lambda - log(beta);
        const double log_lambda = log(lambda);

        uniform_distribution<double> uniform;
        while (true)
        {
            const double u = uniform(g());
            const double x = (alpha - log((1.0 - u) / u)) / beta;
            const double n = floor(x + 0.5);
            if (n < 0)
            {
                continue;
            }
            const double v = uniform(g());
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

    template<class Generator>
    __host__ __device__
    unsigned int generate_huge(Generator& g)
    {
        // Approximate Poisson distribution with normal distribution

        normal_distribution<double> normal;
        const double n = normal(make_uint4(g(), g(), g(), g())).x;
        return static_cast<unsigned int>(round(sqrt(lambda) * n + lambda));
    }

    inline __host__ __device__
    double log_factorial(const double n)
    {
        return (n <= 1.0 ? 0.0 : lgamma(n + 1.0));
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_POISSON_H_
