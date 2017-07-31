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

template<bool IsHostSide = false>
class rocrand_discrete_distribution_poisson : rocrand_discrete_distribution_st
{
public:

    rocrand_discrete_distribution_poisson()
    {
        size = 0;
        probability = NULL;
        alias = NULL;
    }

    rocrand_discrete_distribution_poisson(double lambda)
        : rocrand_discrete_distribution_poisson()
    {
        const size_t capacity =
            2 * static_cast<size_t>(16.0 * (2.0 + std::sqrt(lambda)));
        std::vector<double> p(capacity);

        calculate_probabilities(p, capacity, lambda);
        create_square_histogram(p);
    }

    __host__ __device__
    ~rocrand_discrete_distribution_poisson() { }

    void deallocate()
    {
        // Explicit deallocation is used because on HCC the object is copied
        // multiple times inside hipLaunchKernelGGL, and destructor is called
        // for all copies (we can't use c++ smart pointers for device pointers)
        if (probability != NULL)
        {
            if (IsHostSide)
            {
                delete[] probability;
                delete[] alias;
            }
            else
            {
                hipFree(probability);
                hipFree(alias);
            }
        }
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
        for (int i = capacity / 2 + 1; i < capacity; i++)
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

        size = hi - lo + 1;
        offset = left + lo;
    }

    void allocate()
    {
        if (IsHostSide)
        {
            probability = new double[size];
            alias = new unsigned int[size];
        }
        else
        {
            hipError_t error;
            error = hipMalloc(&probability, sizeof(double) * size);
            if (error != hipSuccess)
            {
                throw ROCRAND_STATUS_ALLOCATION_FAILED;
            }
            error = hipMalloc(&alias, sizeof(unsigned int) * size);
            if (error != hipSuccess)
            {
                throw ROCRAND_STATUS_ALLOCATION_FAILED;
            }
        }
    }

    void create_square_histogram(std::vector<double> p)
    {
        std::vector<double> h_probability(size);
        std::vector<unsigned int> h_alias(size);

        const double average = 1.0 / size;

        double sum = 0.0;
        for (int i = 0; i < size; i++)
        {
            sum += p[i];
        }
        // Normalize probabilities
        for (int i = 0; i < size; i++)
        {
            p[i] /= sum;
        }

        // For detailed descrition of Vose's algorithm see
        // Darts, Dice, and Coins: Sampling from a Discrete Distribution
        // by Keith Schwarz
        // http://www.keithschwarz.com/darts-dice-coins/
        //
        // The algorithm is O(n).

        std::vector<int> small;
        std::vector<int> large;

        small.reserve(size);
        large.reserve(size);

        for (int i = 0; i < size; i++)
        {
            if (p[i] >= average)
                large.push_back(i);
            else
                small.push_back(i);
        }

        while (!small.empty() && !large.empty())
        {
            const int less = small.back();
            small.pop_back();
            const int more = large.back();
            large.pop_back();

            h_probability[less] = p[less] * size;
            h_alias[less] = more;

            p[more] = (p[more] + p[less]) - average;

            if (p[more] >= average)
                large.push_back(more);
            else
                small.push_back(more);
        }

        for (int i : small)
        {
            h_probability[i] = 1.0;
        }
        for (int i : large)
        {
            h_probability[i] = 1.0;
        }

        allocate();

        if (IsHostSide)
        {
            std::copy(h_probability.begin(), h_probability.end(), probability);
            std::copy(h_alias.begin(), h_alias.end(), alias);
        }
        else
        {
            hipError_t error;
            error = hipMemcpy(probability, h_probability.data(), sizeof(double) * size, hipMemcpyDefault);
            if (error != hipSuccess)
            {
                throw ROCRAND_STATUS_INTERNAL_ERROR;
            }
            error = hipMemcpy(alias, h_alias.data(), sizeof(unsigned int) * size, hipMemcpyDefault);
            if (error != hipSuccess)
            {
                throw ROCRAND_STATUS_INTERNAL_ERROR;
            }
        }
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_DISCRETE_H_
