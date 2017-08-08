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

#ifndef ROCRAND_RNG_DISTRIBUTION_UNIFORM_H_
#define ROCRAND_RNG_DISTRIBUTION_UNIFORM_H_

#include <cmath>
#include <hip/hip_runtime.h>

#include "common.hpp"

template<class T>
struct uniform_distribution;

template<>
struct uniform_distribution<unsigned int>
{
    __forceinline__ __host__ __device__
    unsigned int operator()(const unsigned int v) const
    {
        return v;
    }

    __forceinline__ __host__ __device__
    uint4 operator()(const uint4 v) const
    {
        return v;
    }
};

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0f and 1.0f, excluding 0.0f and including 1.0f.
template<>
struct uniform_distribution<float>
{
    __forceinline__ __host__ __device__
    float operator()(const unsigned int v) const
    {
        // return nextafterf(v * ROCRAND_2POW32_INV, 1.0f);
        return v * ROCRAND_2POW32_INV + (ROCRAND_2POW32_INV / 2.0f);
    }

    __forceinline__ __host__ __device__
    float4 operator()(const uint4 v) const
    {
        return {
            (*this)(v.x),
            (*this)(v.y),
            (*this)(v.z),
            (*this)(v.w)
        };
    }
};

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0 and 1.0, excluding 0.0 and including 1.0.
template<>
struct uniform_distribution<double>
{
    __forceinline__ __host__ __device__
    double operator()(const unsigned int v) const
    {
        // return nextafter(v * static_cast<double>(ROCRAND_2POW32_INV), 1.0);
        return v * ROCRAND_2POW32_INV_DOUBLE + (ROCRAND_2POW32_INV_DOUBLE / 2.0);
    }

    __forceinline__ __host__ __device__
    double operator()(const unsigned long long v) const
    {
        return nextafter(
            // 2^53 is the biggest int that can be stored in double, such
            // that it and all smaller integers can be stored in double
            (v >> 11) * ROCRAND_2POW53_INV_DOUBLE, 1.0
        );
    }

    __forceinline__ __host__ __device__
    double4 operator()(const uint4 v) const
    {
        return {
            (*this)(v.x),
            (*this)(v.y),
            (*this)(v.z),
            (*this)(v.w)
        };
    }
};

template<class T>
struct mrg_uniform_distribution;

template<>
struct mrg_uniform_distribution<unsigned int>
{
    __forceinline__ __host__ __device__
    unsigned int operator()(const unsigned long long v) const
    {
        return static_cast<unsigned int>(v);
    }
};

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0f and 1.0f, excluding 0.0f and including 1.0f.
template<>
struct mrg_uniform_distribution<float>
{
    __forceinline__ __host__ __device__
    float operator()(const unsigned long long v) const
    {
        double ret = static_cast<double>(v) * ROCRAND_NORM_DOUBLE;
        return static_cast<float>(ret);
    }
};

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0 and 1.0, excluding 0.0 and including 1.0.
template<>
struct mrg_uniform_distribution<double>
{
    __forceinline__ __host__ __device__
    double operator()(const unsigned long long v) const
    {
        double ret = static_cast<double>(v) * ROCRAND_NORM_DOUBLE;
        return ret;
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_UNIFORM_H_
