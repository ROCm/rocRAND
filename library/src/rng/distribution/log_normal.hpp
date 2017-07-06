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

#include "normal_common.hpp"

template<class T>
struct log_normal_distribution;

template<>
struct log_normal_distribution<float>
{
    float stddev;
    float mean;

    log_normal_distribution<float>(float mean, float stddev) :
                                   mean(mean), stddev(stddev) {}

    __host__ __device__ float2 operator()(unsigned int x, unsigned int y)
    {
        float2 v = box_muller(x, y);
        return expf(mean + (stddev * v));
    }

    __host__ __device__ float4 operator()(uint4 x)
    {
        float2 v = box_muller(x.x, x.y);
        float2 w = box_muller(x.z, x.w);
        return make_float4(expf(mean + (stddev * v)), expf(mean + (stddev * w)));
    }
};

template<>
struct log_normal_distribution<double>
{
    double stddev;
    double mean;

    log_normal_distribution<double>(double mean, double stddev) :
                                    mean(mean), stddev(stddev) {}

    __host__ __device__ double2 operator()(uint4 x)
    {
        double2 v = box_muller_double(x);
        return exp(mean + (stddev * v));
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_LOG_NORMAL_H_
