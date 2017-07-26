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

#include "common.hpp"
#include "normal.hpp"
#include "uniform.hpp"

// George Marsaglia, Wai Wan Tsang, Jingbo Wang
// Fast Generation of Discrete Random Variables
// Journal of Statistical Software, 2004
// https://www.jstatsoft.org/article/view/v011i03

// Only square histogram from Method II is used here (without J[256]).
// Instead of increasing performance, J[256] makes the algorithm slower on GPUs.

namespace rocrand_host {
namespace detail {

template<bool IsHostSide>
class small_poisson_distribution
{
public:

    small_poisson_distribution()
        : size(0), V(NULL), K(NULL) { }

    __host__ __device__
    ~small_poisson_distribution() { }

    void allocate()
    {
        if (V == NULL)
        {
            if (IsHostSide)
            {
                V = new double[capacity];
                K = new unsigned int[capacity];
            }
            else
            {
                hipError_t error;
                error = hipMalloc(&V, sizeof(double) * capacity);
                if (error != hipSuccess)
                {
                    throw ROCRAND_STATUS_ALLOCATION_FAILED;
                }
                error = hipMalloc(&K, sizeof(int) * capacity);
                if (error != hipSuccess)
                {
                    throw ROCRAND_STATUS_ALLOCATION_FAILED;
                }
            }
        }
    }

    void deallocate()
    {
        // Explicit deallocation is used because on HCC the object is copied
        // multiple times inside hipLaunchKernelGGL, and destructor is called
        // for all copies (we can't use c++ smart pointers for device pointers)
        if (V != NULL)
        {
            if (IsHostSide)
            {
                delete[] V;
                delete[] K;
            }
            else
            {
                hipFree(V);
                hipFree(K);
            }
        }
    }

    void set_lambda(double lambda)
    {
        allocate();
        init(lambda);
    }

    template<class Engine>
    __host__ __device__
    unsigned int operator()(Engine& engine) const
    {
        const unsigned int r = engine();
        const double u = r * 2.3283064365386963e-10;
        const unsigned int j = static_cast<unsigned int>(floor(size * u));
        return offset + (u < V[j] ? j : K[j]);
    }

private:

    static constexpr size_t capacity = 1 << 14;

    // Square histogram
    // Top-parts of histogram
    unsigned int * K;
    // Division points (between the bottom-part and the top-part of each column)
    double * V;

    unsigned int size;
    unsigned int offset;

    void init(double lambda)
    {
        std::vector<double> p(capacity);

        calculate_probabilities(p, lambda);
        create_square_histogram(p);
    }

    void calculate_probabilities(std::vector<double>& p, double lambda)
    {
        const double p_epsilon = 1e-9;
        const double log_lambda = std::log(lambda);

        int lo = 0;
        for (int i = static_cast<int>(std::floor(lambda)); i >= 0; i--)
        {
            const double pp = std::exp(i * log_lambda - std::lgamma(i + 1.0) - lambda);
            if (pp < p_epsilon)
            {
                lo = i + 1;
                break;
            }
            p[i] = pp;
        }

        int hi = capacity - 1;
        for (int i = static_cast<int>(std::ceil(lambda)); i < capacity; i++)
        {
            const double pp = std::exp(i * log_lambda - std::lgamma(i + 1.0) - lambda);
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

        size = hi - lo + 1;
        offset = lo;
    }

    void create_square_histogram(std::vector<double>& p)
    {
        std::vector<double> h_V(size);
        std::vector<unsigned int> h_K(size);

        const double a = 1.0 / size;
        for (int i = 0; i < size; i++)
        {
            h_K[i] = i;
            h_V[i] = (i + 1) * a;
        }

        // Apply Robin Hood rule size - 1 times:
        for (int s = 0; s < size - 1; s++)
        {
            // Find the minimum and maximum columns
            double min_p = a;
            double max_p = a;
            int min_i = 0;
            int max_i = 0;
            for (int i = 0; i < size; i++)
            {
                const double t = p[i];
                if (t < min_p)
                {
                    min_p = t;
                    min_i = i;
                }
                if (t > max_p)
                {
                    max_p = t;
                    max_i = i;
                }
            }
            // Store donating index in K[] and division point in V[]
            h_V[min_i] = min_p + min_i * a;
            h_K[min_i] = max_i;
            // Take from maximum to bring minimum to average.
            p[max_i] = max_p + min_p - a;
            p[min_i] = a;
        }

        if (IsHostSide)
        {
            std::copy(h_V.begin(), h_V.end(), V);
            std::copy(h_K.begin(), h_K.end(), K);
        }
        else
        {
            hipError_t error;
            error = hipMemcpy(V, h_V.data(), sizeof(double) * size, hipMemcpyDefault);
            if (error != hipSuccess)
            {
                throw ROCRAND_STATUS_INTERNAL_ERROR;
            }
            error = hipMemcpy(K, h_K.data(), sizeof(unsigned int) * size, hipMemcpyDefault);
            if (error != hipSuccess)
            {
                throw ROCRAND_STATUS_INTERNAL_ERROR;
            }
        }
    }
};

template<bool IsHostSide>
class large_poisson_distribution
{
public:

    large_poisson_distribution() { }

    __host__ __device__
    ~large_poisson_distribution() { }

    template<class Engine>
    __host__ __device__
    uint2 operator()(Engine& engine) const
    {
        // Approximate Poisson distribution with normal distribution
        normal_distribution<double> normal(lambda, sqrt_lambda);
        const double2 n = normal(
            uint4 { engine(), engine(), engine(), engine() }
        );
        return uint2 {
            static_cast<unsigned int>(round(n.x)),
            static_cast<unsigned int>(round(n.y))
        };
    }

    void set_lambda(double new_lambda)
    {
        lambda = new_lambda;
        sqrt_lambda = sqrt(new_lambda);
    }

private:

    double lambda;
    double sqrt_lambda;
};


} // end namespace detail
} // end namespace rocrand_host

template<bool IsHostSide = false>
class poisson_distribution
{
public:

    rocrand_host::detail::small_poisson_distribution<IsHostSide> small;
    rocrand_host::detail::large_poisson_distribution<IsHostSide> large;

    poisson_distribution()
        : lambda(0.0)
    { }

    ~poisson_distribution()
    {
        small.deallocate();
    }

    void set_lambda(double new_lambda)
    {
        const bool changed = lambda != new_lambda;
        if (changed)
        {
            lambda = new_lambda;
            if (use_small())
            {
                small.set_lambda(lambda);
            }
            else
            {
                large.set_lambda(lambda);
            }
        }
    }

    bool use_small()
    {
        return lambda < lambda_threshold_small;
    }

private:

    static constexpr double lambda_threshold_small = 4000.0;

    double lambda;
};


template<class T>
class mrg_poisson_distribution;

template<>
class mrg_poisson_distribution<unsigned int>
{
public:

    __host__ __device__
    mrg_poisson_distribution(double lambda)
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

        mrg_uniform_distribution<double> uniform;
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
        const double beta = ROCRAND_PI_DOUBLE / sqrt(3.0 * lambda);
        const double alpha = beta * lambda;
        const double k = log(c) - lambda - log(beta);
        const double log_lambda = log(lambda);

        mrg_uniform_distribution<double> uniform;
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

        mrg_normal_distribution<double> normal;
        const double n = normal(g(), g()).x;
        return static_cast<unsigned int>(round(sqrt(lambda) * n + lambda));
    }

    inline __host__ __device__
    double log_factorial(const double n)
    {
        return (n <= 1.0 ? 0.0 : lgamma(n + 1.0));
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_POISSON_H_
