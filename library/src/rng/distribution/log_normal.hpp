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

#ifndef ROCRAND_RNG_DISTRIBUTION_LOG_NORMAL_H_
#define ROCRAND_RNG_DISTRIBUTION_LOG_NORMAL_H_

#include <cmath>
#include <hip/hip_runtime.h>

#include "common.hpp"
#include "normal_common.hpp"
#include "uniform.hpp"

template<class T>
struct log_normal_distribution;

template<>
struct log_normal_distribution<float>
{
    const float mean;
    const float stddev;

    __host__ __device__
    log_normal_distribution<float>(const float mean, const float stddev) :
                                   mean(mean), stddev(stddev) {}

    __forceinline__ __host__ __device__
    float2 operator()(const unsigned int x, const unsigned int y)
    {
        float2 v = box_muller(x, y);
        v.x = expf(mean + (stddev * v.x));
        v.y = expf(mean + (stddev * v.y));
        return v;
    }

    __forceinline__ __host__ __device__
    float4 operator()(const uint4 x)
    {
        float2 v = box_muller(x.x, x.y);
        float2 w = box_muller(x.z, x.w);
        return float4 {
            expf(mean + (stddev * v.x)),
            expf(mean + (stddev * v.y)),
            expf(mean + (stddev * w.x)),
            expf(mean + (stddev * w.y))
        };
    }
    
    // inverse CDF
    // TODO: find alternative as performance is low
    __forceinline__ __host__ __device__
    float operator()(unsigned int x)
    {
        uniform_distribution<float> uniform;
        float p = uniform(x);
        float v;
        if (p < 0.5f)
            v = -inverse_f_cdf(sqrtf(-2.0f * logf(p)));
        else
            v =  inverse_f_cdf(sqrtf(-2.0f * logf(1.0f - p)));
        v = expf(mean + (stddev * v));
        return v;
    }
};

template<>
struct log_normal_distribution<double>
{
    const double mean;
    const double stddev;

    __host__ __device__
    log_normal_distribution<double>(const double mean, const double stddev) :
                                    mean(mean), stddev(stddev) {}

    __forceinline__ __host__ __device__
    double2 operator()(const uint4 x)
    {
        double2 v = box_muller_double(x);
        v.x = exp(mean + (stddev * v.x));
        v.y = exp(mean + (stddev * v.y));
        return v;
    }
    
    // inverse CDF
    // TODO: find alternative as performance is low
    __forceinline__ __host__ __device__
    double operator()(unsigned int x)
    {
        uniform_distribution<double> uniform;
        double p = uniform(x);
        double v;
        if (p < 0.5)
            v = -inverse_d_cdf(sqrt(-2.0 * log(p)));
        else
            v =  inverse_d_cdf(sqrt(-2.0 * log(1.0 - p)));
        v = exp(mean + (stddev * v));
        return v;
    }
};

template<class T>
struct mrg_log_normal_distribution;

template<>
struct mrg_log_normal_distribution<float>
{
    const float mean;
    const float stddev;

    __host__ __device__
    mrg_log_normal_distribution<float>(float mean = 0.0f, float stddev = 1.0f) :
                                       mean(mean), stddev(stddev) {}

    __forceinline__ __host__ __device__
    float2 operator()(const unsigned long long x, const unsigned long long y)
    {
        mrg_uniform_distribution<float> uniform;
        float a = uniform(x);
        float b = uniform(y);
        float2 v = box_muller_mrg(a, b);
        v.x = exp(mean + (stddev * v.x));
        v.y = exp(mean + (stddev * v.y));
        return v;
    }
};

template<>
struct mrg_log_normal_distribution<double>
{
    const double mean;
    const double stddev;

    __host__ __device__
    mrg_log_normal_distribution<double>(double mean = 0.0, double stddev = 1.0) :
                                        mean(mean), stddev(stddev) {}

    __forceinline__ __host__ __device__
    double2 operator()(const unsigned long long x, const unsigned long long y)
    {
        mrg_uniform_distribution<double> uniform;
        double a = uniform(x);
        double b = uniform(y);
        double2 v = box_muller_double_mrg(a, b);
        v.x = exp(mean + (stddev * v.x));
        v.y = exp(mean + (stddev * v.y));
        return v;
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_LOG_NORMAL_H_
