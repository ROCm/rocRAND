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
        return ROCRAND_2POW32_INV + (v * ROCRAND_2POW32_INV);
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
    struct two_uints
    {
        unsigned int x;
        unsigned int y;
    };

    union two_uints_to_ulong
    {
        two_uints uint2_value;
        unsigned long long ulong_value;
    };

    __forceinline__ __host__ __device__
    double operator()(const unsigned int v) const
    {
        return ROCRAND_2POW32_INV_DOUBLE + (v * ROCRAND_2POW32_INV_DOUBLE);
    }

    __forceinline__ __host__ __device__
    double operator()(const unsigned int v1, const unsigned int v2) const
    {
        two_uints_to_ulong v;
        v.uint2_value.x = v1;
        v.uint2_value.y = (v2 >> 11);
        return ROCRAND_2POW53_INV_DOUBLE + (v.ulong_value * ROCRAND_2POW53_INV_DOUBLE);
    }

    __forceinline__ __host__ __device__
    double operator()(const unsigned long long v) const
    {
        return ROCRAND_2POW53_INV_DOUBLE + (
            // 2^53 is the biggest int that can be stored in double, such
            // that it and all smaller integers can be stored in double
            (v >> 11) * ROCRAND_2POW53_INV_DOUBLE
        );
    }

    __forceinline__ __host__ __device__
    double2 operator()(const uint4 v) const
    {
        return double2 {
            (*this)(v.x, v.y),
            (*this)(v.z, v.w)
        };
    }

    __forceinline__ __host__ __device__
    double4 operator()(const uint4 v1, const uint4 v2) const
    {
        return double4 {
            (*this)(v1.x, v1.y),
            (*this)(v1.z, v1.w),
            (*this)(v2.x, v2.y),
            (*this)(v2.z, v2.w)
        };
    }
};

// MRG32K3A constants
#ifndef ROCRAND_MRG32K3A_NORM_DOUBLE
#define ROCRAND_MRG32K3A_NORM_DOUBLE (2.3283065498378288e-10) // 1/ROCRAND_MRG32K3A_M1
#endif
#ifndef ROCRAND_MRG32K3A_UINT_NORM
#define ROCRAND_MRG32K3A_UINT_NORM (1.000000048661606966) // ROCRAND_MRG32K3A_POW32/ROCRAND_MRG32K3A_M1
#endif

template<class T>
struct mrg_uniform_distribution;

template<>
struct mrg_uniform_distribution<unsigned int>
{
    __forceinline__ __host__ __device__
    unsigned int operator()(const unsigned int v) const
    {
        return static_cast<unsigned int>(v * ROCRAND_MRG32K3A_UINT_NORM);
    }
};

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0f and 1.0f, excluding 0.0f and including 1.0f.
template<>
struct mrg_uniform_distribution<float>
{
    __forceinline__ __host__ __device__
    float operator()(const unsigned int v) const
    {
        double ret = static_cast<double>(v) * ROCRAND_MRG32K3A_NORM_DOUBLE;
        return static_cast<float>(ret);
    }
};

// For unsigned integer between 0 and UINT_MAX, returns value between
// 0.0 and 1.0, excluding 0.0 and including 1.0.
template<>
struct mrg_uniform_distribution<double>
{
    __forceinline__ __host__ __device__
    double operator()(const unsigned int v) const
    {
        double ret = static_cast<double>(v) * ROCRAND_MRG32K3A_NORM_DOUBLE;
        return ret;
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_UNIFORM_H_
