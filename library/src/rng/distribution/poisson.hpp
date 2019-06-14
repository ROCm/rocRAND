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
#include <algorithm>
#include <vector>

#include <rocrand.h>

#include "discrete.hpp"

template<rocrand_discrete_method Method = ROCRAND_DISCRETE_METHOD_ALIAS, bool IsHostSide = false>
class rocrand_poisson_distribution : public rocrand_discrete_distribution_base<Method, IsHostSide>
{
public:

    typedef rocrand_discrete_distribution_base<Method, IsHostSide> base;

    rocrand_poisson_distribution()
        : base() { }

    rocrand_poisson_distribution(double lambda)
        : rocrand_poisson_distribution()
    {
        set_lambda(lambda);
    }

    __host__ __device__
    ~rocrand_poisson_distribution() { }

    void set_lambda(double lambda)
    {
        const size_t capacity =
            2 * static_cast<size_t>(16.0 * (2.0 + std::sqrt(lambda)));
        std::vector<double> p(capacity);

        calculate_probabilities(p, capacity, lambda);

        this->init(p, this->size, this->offset);
    }

protected:

    void calculate_probabilities(std::vector<double>& p, const size_t capacity,
                                 const double lambda)
    {
        const double p_epsilon = 1e-12;
        const double log_lambda = std::log(lambda);

        const int left = static_cast<int>(std::floor(lambda)) - capacity / 2;

        // Calculate probabilities starting from mean in both directions,
        // because only a small part of [0, lambda] has non-negligible values
        // (> p_epsilon).

        int lo = 0;
        for (int i = capacity / 2; i >= 0; i--)
        {
            const double x = left + i;
            const double pp = std::exp(x * log_lambda - std::lgamma(x + 1.0) - lambda);
            if (pp < p_epsilon)
            {
                lo = i + 1;
                break;
            }
            p[i] = pp;
        }

        int hi = capacity - 1;
        for (int i = capacity / 2 + 1; i < static_cast<int>(capacity); i++)
        {
            const double x = left + i;
            const double pp = std::exp(x * log_lambda - std::lgamma(x + 1.0) - lambda);
            if (pp < p_epsilon)
            {
                hi = i - 1;
                break;
            }
            p[i] = pp;
        }

        for (int i = lo; i <= hi; i++)
        {
            p[i - lo] = p[i];
        }

        this->size = hi - lo + 1;
        this->offset = left + lo;
    }
};

// Handles caching of precomputed tables for the distribution and recomputes
// them only when lambda is changed (as these computations, device memory
// allocations and copying take time).
template<rocrand_discrete_method Method = ROCRAND_DISCRETE_METHOD_ALIAS, bool IsHostSide = false>
class poisson_distribution_manager
{
public:

    rocrand_poisson_distribution<Method, IsHostSide> dis;

    poisson_distribution_manager()
        : lambda(0.0)
    { }

    ~poisson_distribution_manager()
    {
        dis.deallocate();
    }

    void set_lambda(double new_lambda)
    {
        const bool changed = lambda != new_lambda;
        if (changed)
        {
            lambda = new_lambda;
            dis.set_lambda(lambda);
        }
    }

private:

    double lambda;
};

#endif // ROCRAND_RNG_DISTRIBUTION_POISSON_H_
