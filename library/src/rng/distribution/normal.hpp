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

#include "device_distributions.hpp"


// Universal

template<class T>
struct normal_distribution;

template<>
struct normal_distribution<float>
{
    static constexpr unsigned int input_width = 2;
    static constexpr unsigned int output_width = 2;

    const float mean;
    const float stddev;

    __host__ __device__
    normal_distribution(float mean, float stddev)
        : mean(mean), stddev(stddev) {}

    __host__ __device__
    void operator()(const unsigned int (&input)[2], float (&output)[2]) const
    {
        float2 v = rocrand_device::detail::normal_distribution2(input[0], input[1]);
        output[0] = mean + v.x * stddev;
        output[1] = mean + v.y * stddev;
    }
};

template<>
struct normal_distribution<double>
{
    static constexpr unsigned int input_width = 4;
    static constexpr unsigned int output_width = 2;

    const double mean;
    const double stddev;

    __host__ __device__
    normal_distribution(double mean, double stddev)
        : mean(mean), stddev(stddev) {}

    __host__ __device__
    void operator()(const unsigned int (&input)[4], double (&output)[2]) const
    {
        double2 v = rocrand_device::detail::normal_distribution_double2(
            make_uint4(input[0], input[1], input[2], input[3])
        );
        output[0] = mean + v.x * stddev;
        output[1] = mean + v.y * stddev;
    }
};

template<>
struct normal_distribution<__half>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 2;

    const __half2 mean;
    const __half2 stddev;

    __host__ __device__
    normal_distribution(__half mean, __half stddev)
        : mean(mean, mean), stddev(stddev, stddev) {}

    __host__ __device__
    void operator()(const unsigned int (&input)[1], __half (&output)[2]) const
    {
        __half2 v = rocrand_device::detail::normal_distribution_half2(input[0]);
        #if defined(ROCRAND_HALF_MATH_SUPPORTED)
        *reinterpret_cast<__half2 *>(output) = __hfma2(v, stddev, mean);
        #else
        output[0] = __float2half(__low2float(mean) + (__low2float(stddev) * __low2float(v)));
        output[1] = __float2half(__low2float(mean) + (__low2float(stddev) * __high2float(v)));
        #endif
    }
};


// Mrg32k3a

template<class T>
struct mrg_normal_distribution;

template<>
struct mrg_normal_distribution<float>
{
    static constexpr unsigned int input_width = 2;
    static constexpr unsigned int output_width = 2;

    const float mean;
    const float stddev;

    __host__ __device__
    mrg_normal_distribution(float mean, float stddev)
        : mean(mean), stddev(stddev) {}

    __host__ __device__
    void operator()(const unsigned int (&input)[2], float (&output)[2]) const
    {
        float2 v = rocrand_device::detail::mrg_normal_distribution2(input[0], input[1]);
        output[0] = mean + v.x * stddev;
        output[1] = mean + v.y * stddev;
    }
};

template<>
struct mrg_normal_distribution<double>
{
    static constexpr unsigned int input_width = 2;
    static constexpr unsigned int output_width = 2;

    const double mean;
    const double stddev;

    __host__ __device__
    mrg_normal_distribution(double mean, double stddev)
        : mean(mean), stddev(stddev) {}

    __host__ __device__
    void operator()(const unsigned int (&input)[2], double (&output)[2]) const
    {
        double2 v = rocrand_device::detail::mrg_normal_distribution_double2(input[0], input[1]);
        output[0] = mean + v.x * stddev;
        output[1] = mean + v.y * stddev;
    }
};

template<>
struct mrg_normal_distribution<__half>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 2;

    const __half2 mean;
    const __half2 stddev;

    __host__ __device__
    mrg_normal_distribution(__half mean, __half stddev)
        : mean(mean, mean), stddev(stddev, stddev) {}

    __host__ __device__
    void operator()(const unsigned int (&input)[1], __half (&output)[2]) const
    {
        __half2 v = rocrand_device::detail::mrg_normal_distribution_half2(input[0]);
        #if defined(ROCRAND_HALF_MATH_SUPPORTED)
        *reinterpret_cast<__half2 *>(output) = __hfma2(v, stddev, mean);
        #else
        output[0] = __float2half(__low2float(mean) + (__low2float(stddev) * __low2float(v)));
        output[1] = __float2half(__low2float(mean) + (__low2float(stddev) * __high2float(v)));
        #endif
    }
};


// Sobol

template<class T>
struct sobol_normal_distribution;

template<>
struct sobol_normal_distribution<float>
{
    const float mean;
    const float stddev;

    __host__ __device__
    sobol_normal_distribution(float mean, float stddev)
        : mean(mean), stddev(stddev) {}

    __host__ __device__
    float operator()(const unsigned int x) const
    {
        float v = rocrand_device::detail::normal_distribution(x);
        return mean + v * stddev;
    }
};

template<>
struct sobol_normal_distribution<double>
{
    const double mean;
    const double stddev;

    __host__ __device__
    sobol_normal_distribution(double mean, double stddev)
        : mean(mean), stddev(stddev) {}

    __host__ __device__
    double operator()(const unsigned int x) const
    {
        double v = rocrand_device::detail::normal_distribution_double(x);
        return mean + v * stddev;
    }
};

template<>
struct sobol_normal_distribution<__half>
{
    const __half mean;
    const __half stddev;

    __host__ __device__
    sobol_normal_distribution(__half mean, __half stddev)
        : mean(mean), stddev(stddev) {}

    __host__ __device__
    __half operator()(const unsigned int x) const
    {
        float v = rocrand_device::detail::normal_distribution(x);
        #if defined(ROCRAND_HALF_MATH_SUPPORTED)
        return __hfma(__float2half(v), stddev, mean);
        #else
        return __half2float(mean) + v * __half2float(stddev);
        #endif
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_NORMAL_H_
