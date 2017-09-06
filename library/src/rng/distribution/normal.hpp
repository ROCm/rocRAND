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

#ifndef ROCRAND_RNG_DISTRIBUTION_NORMAL_H_
#define ROCRAND_RNG_DISTRIBUTION_NORMAL_H_

#include <math.h>
#include <hip/hip_runtime.h>

#include "common.hpp"
#include "device_distributions.hpp"

template<class T>
struct normal_distribution;

template<>
struct normal_distribution<float>
{
    const float mean;
    const float stddev;

    __host__ __device__
    normal_distribution<float>(float mean = 0.0f, float stddev = 1.0f) :
                               mean(mean), stddev(stddev) {}

    __forceinline__ __host__ __device__
    float2 operator()(const unsigned int x, const unsigned int y)
    {
        float2 v = rocrand_device::detail::box_muller(x, y);
        v.x = mean + v.x * stddev;
        v.y = mean + v.y * stddev;
        return v;
    }

    __forceinline__ __host__ __device__
    float2 operator()(const uint2 x)
    {
        float2 v = rocrand_device::detail::box_muller(x.x, x.y);
        v.x = mean + v.x * stddev;
        v.y = mean + v.y * stddev;
        return v;
    }

    __forceinline__ __host__ __device__
    float4 operator()(const uint4 x)
    {
        float2 v = rocrand_device::detail::box_muller(x.x, x.y);
        float2 w = rocrand_device::detail::box_muller(x.z, x.w);
        return float4{
            mean + v.x * stddev,
            mean + v.y * stddev,
            mean + w.x * stddev,
            mean + w.y * stddev,
        };
    }

    __forceinline__ __host__ __device__
    float operator()(const unsigned int x)
    {
        float v = rocrand_device::detail::normal_distribution(x);
        return mean + v * stddev;
    }
};

template<>
struct normal_distribution<double>
{
    const double mean;
    const double stddev;

    __host__ __device__
    normal_distribution<double>(double mean = 0.0, double stddev = 1.0) :
                                mean(mean), stddev(stddev) {}

    __forceinline__ __host__ __device__
    double2 operator()(uint4 x)
    {
        double2 v = rocrand_device::detail::box_muller_double(x);
        v.x = mean + v.x * stddev;
        v.y = mean + v.y * stddev;
        return v;
    }

    __forceinline__ __host__ __device__
    double operator()(const unsigned int x)
    {
        double v = rocrand_device::detail::normal_distribution_double(x);
        return mean + v * stddev;
    }
};

template<class T>
struct mrg_normal_distribution;

template<>
struct mrg_normal_distribution<float>
{
    const float mean;
    const float stddev;

    __host__ __device__
    mrg_normal_distribution<float>(float mean = 0.0f, float stddev = 1.0f) :
                                   mean(mean), stddev(stddev) {}

    __forceinline__ __host__ __device__
    float2 operator()(const unsigned int x, const unsigned int y)
    {
        float2 v = rocrand_device::detail::mrg_normal_distribution2(x, y);
        v.x = mean + v.x * stddev;
        v.y = mean + v.y * stddev;
        return v;
    }
};

template<>
struct mrg_normal_distribution<double>
{
    const double mean;
    const double stddev;

    __host__ __device__
    mrg_normal_distribution<double>(double mean = 0.0, double stddev = 1.0) :
                                    mean(mean), stddev(stddev) {}

    __forceinline__ __host__ __device__
    double2 operator()(const unsigned int x, const unsigned int y)
    {
        double2 v = rocrand_device::detail::mrg_normal_distribution_double2(x, y);
        v.x = mean + v.x * stddev;
        v.y = mean + v.y * stddev;
        return v;
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_NORMAL_H_
