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

#ifndef ROCRAND_RNG_DISTRIBUTION_DISCRETE_H_
#define ROCRAND_RNG_DISTRIBUTION_DISCRETE_H_

#include <climits>
#include <algorithm>
#include <vector>

#include <rocrand.h>

// George Marsaglia, Wai Wan Tsang, Jingbo Wang
// Fast Generation of Discrete Random Variables
// Journal of Statistical Software, 2004
// https://www.jstatsoft.org/article/view/v011i03

// Only square histogram from Method II is used here (without J[256]).
// Instead of increasing performance, J[256] makes the algorithm slower on GPUs.

template<bool IsHostSide = false>
class rocrand_discrete_distribution_poisson : rocrand_discrete_distribution_st
{
public:

    rocrand_discrete_distribution_poisson()
    {
        size = 0;
        V = NULL;
        K = NULL;
    }

    rocrand_discrete_distribution_poisson(double lambda)
        : rocrand_discrete_distribution_poisson()
    {
        normal_mean = lambda;
        normal_stddev = std::sqrt(lambda);

        if (lambda < lambda_threshold_small)
        {
            const size_t capacity =
                static_cast<size_t>(lambda + 16.0 * (2.0 + std::sqrt(lambda)));
            std::vector<double> p(capacity);

            calculate_probabilities(p, capacity, lambda);
            create_square_histogram(p);
        }
    }

    __host__ __device__
    ~rocrand_discrete_distribution_poisson() { }

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

protected:

    static constexpr double lambda_threshold_small = 2000.0;

    void calculate_probabilities(std::vector<double>& p, size_t capacity, double lambda)
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
        for (int i = static_cast<int>(std::floor(lambda)) + 1; i < capacity; i++)
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

        // Normalize probailities
        double sum = 0.0;
        for (int i = 0; i < size; i++)
        {
            sum += p[i];
        }
        for (int i = 0; i < size; i++)
        {
            p[i] /= sum;
        }

        // Apply Robin Hood rule size - 1 times:
        for (int s = 0; s < size - 1; s++)
        {
            // Find the minimum and maximum columns
            double min_p = a;
            double max_p = a;
            int min_i = -1;
            int max_i = -1;
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
            if (min_i != -1 && max_i != -1)
            {
                // Store donating index in K[] and division point in V[]
                h_V[min_i] = min_p + min_i * a;
                h_K[min_i] = max_i;
                // Take from maximum to bring minimum to average.
                p[max_i] = max_p - (a - min_p);
                p[min_i] = a;
            }
        }

        if (IsHostSide)
        {
            V = new double[size];
            K = new unsigned int[size];

            std::copy(h_V.begin(), h_V.end(), V);
            std::copy(h_K.begin(), h_K.end(), K);
        }
        else
        {
            hipError_t error;

            error = hipMalloc(&V, sizeof(double) * size);
            if (error != hipSuccess)
            {
                throw ROCRAND_STATUS_ALLOCATION_FAILED;
            }
            error = hipMalloc(&K, sizeof(unsigned int) * size);
            if (error != hipSuccess)
            {
                throw ROCRAND_STATUS_ALLOCATION_FAILED;
            }

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

#endif // ROCRAND_RNG_DISTRIBUTION_DISCRETE_H_
