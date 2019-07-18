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

#include "device_distributions.hpp"

// Alias method
//
// Walker, A. J.
// An Efficient Method for Generating Discrete Random Variables with General Distributions, 1977
//
// Vose M. D.
// A Linear Algorithm For Generating Random Numbers With a Given Distribution, 1991

enum rocrand_discrete_method
{
    ROCRAND_DISCRETE_METHOD_ALIAS = 1,
    ROCRAND_DISCRETE_METHOD_CDF = 2,
    ROCRAND_DISCRETE_METHOD_UNIVERSAL = ROCRAND_DISCRETE_METHOD_ALIAS | ROCRAND_DISCRETE_METHOD_CDF
};

template<rocrand_discrete_method Method = ROCRAND_DISCRETE_METHOD_ALIAS, bool IsHostSide = false>
class rocrand_discrete_distribution_base : public rocrand_discrete_distribution_st
{
public:

    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 1;

    rocrand_discrete_distribution_base()
    {
        size = 0;
        probability = NULL;
        alias = NULL;
        cdf = NULL;
    }

    rocrand_discrete_distribution_base(const double * probabilities,
                                       unsigned int size,
                                       unsigned int offset)
        : rocrand_discrete_distribution_base()
    {
        std::vector<double> p(probabilities, probabilities + size);

        init(p, size, offset);
    }

    __host__ __device__
    ~rocrand_discrete_distribution_base() { }

    void deallocate()
    {
        // Explicit deallocation is used because on HCC the object is copied
        // multiple times inside hipLaunchKernelGGL, and destructor is called
        // for all copies (we can't use c++ smart pointers for device pointers)
        if (IsHostSide)
        {
            if (probability != NULL)
            {
                delete[] probability;
            }
            if (alias != NULL)
            {
                delete[] alias;
            }
            if (cdf != NULL)
            {
                delete[] cdf;
            }
        }
        else
        {
            if (probability != NULL)
            {
                hipFree(probability);
            }
            if (alias != NULL)
            {
                hipFree(alias);
            }
            if (cdf != NULL)
            {
                hipFree(cdf);
            }
        }
        probability = NULL;
        alias = NULL;
        cdf = NULL;
    }

    __forceinline__ __host__ __device__
    unsigned int operator()(const unsigned int x) const
    {
        if ((Method & ROCRAND_DISCRETE_METHOD_ALIAS) != 0)
        {
            return rocrand_device::detail::discrete_alias(x, *this);
        }
        else
        {
            return rocrand_device::detail::discrete_cdf(x, *this);
        }
    }

    __host__ __device__
    void operator()(const unsigned int (&input)[1], unsigned int output[1]) const
    {
        output[0] = (*this)(input[0]);
    }

protected:

    void init(std::vector<double> p,
              const unsigned int size,
              const unsigned int offset)
    {
        this->size = size;
        this->offset = offset;

        deallocate();
        allocate();
        normalize(p);
        if ((Method & ROCRAND_DISCRETE_METHOD_ALIAS) != 0)
        {
            create_alias_table(p);
        }
        if ((Method & ROCRAND_DISCRETE_METHOD_CDF) != 0)
        {
            create_cdf(p);
        }
    }

    void allocate()
    {
        if (IsHostSide)
        {
            if ((Method & ROCRAND_DISCRETE_METHOD_ALIAS) != 0)
            {
                probability = new double[size];
                alias = new unsigned int[size];
            }
            if ((Method & ROCRAND_DISCRETE_METHOD_CDF) != 0)
            {
                cdf = new double[size];
            }
        }
        else
        {
            hipError_t error;
            if ((Method & ROCRAND_DISCRETE_METHOD_ALIAS) != 0)
            {
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
            if ((Method & ROCRAND_DISCRETE_METHOD_CDF) != 0)
            {
                error = hipMalloc(&cdf, sizeof(double) * size);
                if (error != hipSuccess)
                {
                    throw ROCRAND_STATUS_ALLOCATION_FAILED;
                }
            }
        }
    }

    void normalize(std::vector<double>& p)
    {
        double sum = 0.0;
        for (unsigned int i = 0; i < size; i++)
        {
            sum += p[i];
        }
        // Normalize probabilities
        for (unsigned int i = 0; i < size; i++)
        {
            p[i] /= sum;
        }
    }

    void create_alias_table(std::vector<double> p)
    {
        std::vector<double> h_probability(size);
        std::vector<unsigned int> h_alias(size);

        const double average = 1.0 / size;

        // For detailed descrition of Vose's algorithm see
        // Darts, Dice, and Coins: Sampling from a Discrete Distribution
        // by Keith Schwarz
        // http://www.keithschwarz.com/darts-dice-coins/
        //
        // The algorithm is O(n).

        std::vector<unsigned int> small;
        std::vector<unsigned int> large;

        small.reserve(size);
        large.reserve(size);

        for (unsigned int i = 0; i < size; i++)
        {
            if (p[i] >= average)
                large.push_back(i);
            else
                small.push_back(i);
        }

        while (!small.empty() && !large.empty())
        {
            const unsigned int less = small.back();
            small.pop_back();
            const unsigned int more = large.back();
            large.pop_back();

            h_probability[less] = p[less] * size;
            h_alias[less] = more;

            p[more] = (p[more] + p[less]) - average;

            if (p[more] >= average)
                large.push_back(more);
            else
                small.push_back(more);
        }

        for (unsigned int i : small)
        {
            h_probability[i] = 1.0;
        }
        for (unsigned int i : large)
        {
            h_probability[i] = 1.0;
        }

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

    void create_cdf(std::vector<double> p)
    {
        std::vector<double> h_cdf(size);

        double sum = 0.0;
        for (unsigned int i = 0; i < size; i++)
        {
            sum += p[i];
            h_cdf[i] = sum;
        }

        if (IsHostSide)
        {
            std::copy(h_cdf.begin(), h_cdf.end(), cdf);
        }
        else
        {
            hipError_t error;
            error = hipMemcpy(cdf, h_cdf.data(), sizeof(double) * size, hipMemcpyDefault);
            if (error != hipSuccess)
            {
                throw ROCRAND_STATUS_INTERNAL_ERROR;
            }
        }
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_DISCRETE_H_
