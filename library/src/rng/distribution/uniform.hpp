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

#include <math.h>
#include <hip/hip_runtime.h>

#include "device_distributions.hpp"


// Universal

template<class T>
struct uniform_distribution;

template<>
struct uniform_distribution<unsigned int>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 1;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], unsigned int (&output)[1]) const
    {
        unsigned int v = input[0];
        output[0] = v;
    }
};

template<>
struct uniform_distribution<unsigned char>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 4;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], unsigned char (&output)[4]) const
    {
        unsigned int v = input[0];
        *reinterpret_cast<unsigned int *>(output) = v;
    }
};

template<>
struct uniform_distribution<unsigned short>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 2;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], unsigned short (&output)[2]) const
    {
        unsigned int v = input[0];
        *reinterpret_cast<unsigned int *>(output) = v;
    }
};

template<>
struct uniform_distribution<float>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 1;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], float (&output)[1]) const
    {
        output[0] = rocrand_device::detail::uniform_distribution(input[0]);
    }
};

template<>
struct uniform_distribution<double>
{
    static constexpr unsigned int input_width = 2;
    static constexpr unsigned int output_width = 1;

    __host__ __device__
    void operator()(const unsigned int (&input)[2], double (&output)[1]) const
    {
        output[0] = rocrand_device::detail::uniform_distribution_double(input[0], input[1]);
    }
};

template<>
struct uniform_distribution<__half>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 2;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], __half (&output)[2]) const
    {
        unsigned int v = input[0];
        output[0] = rocrand_device::detail::uniform_distribution_half(static_cast<short>(v));
        output[1] = rocrand_device::detail::uniform_distribution_half(static_cast<short>(v >> 16));
    }
};


// Mrg32k3a

template<class T>
struct mrg_uniform_distribution;

template<>
struct mrg_uniform_distribution<unsigned int>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 1;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], unsigned int (&output)[1]) const
    {
        unsigned int v = rocrand_device::detail::mrg_uniform_distribution_uint(input[0]);
        output[0] = v;
    }
};

template<>
struct mrg_uniform_distribution<unsigned char>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 4;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], unsigned char (&output)[4]) const
    {
        unsigned int v = rocrand_device::detail::mrg_uniform_distribution_uint(input[0]);
        *reinterpret_cast<unsigned int *>(output) = v;
    }
};

template<>
struct mrg_uniform_distribution<unsigned short>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 2;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], unsigned short (&output)[2]) const
    {
        unsigned int v = rocrand_device::detail::mrg_uniform_distribution_uint(input[0]);
        *reinterpret_cast<unsigned int *>(output) = v;
    }
};

template<>
struct mrg_uniform_distribution<float>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 1;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], float (&output)[1]) const
    {
        output[0] = rocrand_device::detail::mrg_uniform_distribution(input[0]);
    }
};

template<>
struct mrg_uniform_distribution<double>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 1;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], double (&output)[1]) const
    {
        output[0] = rocrand_device::detail::mrg_uniform_distribution_double(input[0]);
    }
};

template<>
struct mrg_uniform_distribution<__half>
{
    static constexpr unsigned int input_width = 1;
    static constexpr unsigned int output_width = 2;

    __host__ __device__
    void operator()(const unsigned int (&input)[1], __half (&output)[2]) const
    {
        unsigned int v = rocrand_device::detail::mrg_uniform_distribution_uint(input[0]);
        output[0] = rocrand_device::detail::uniform_distribution_half(static_cast<short>(v));
        output[1] = rocrand_device::detail::uniform_distribution_half(static_cast<short>(v >> 16));
    }
};


// Sobol

template<class T>
struct sobol_uniform_distribution;

template<>
struct sobol_uniform_distribution<unsigned int>
{
    __host__ __device__
    unsigned int operator()(const unsigned int v) const
    {
        return v;
    }
};

template<>
struct sobol_uniform_distribution<unsigned char>
{
    __host__ __device__
    unsigned char operator()(const unsigned int v) const
    {
        return static_cast<unsigned char>(v >> 24);
    }
};

template<>
struct sobol_uniform_distribution<unsigned short>
{
    __host__ __device__
    unsigned short operator()(const unsigned int v) const
    {
        return static_cast<unsigned short>(v >> 16);
    }
};

template<>
struct sobol_uniform_distribution<float>
{
    __host__ __device__
    float operator()(const unsigned int v) const
    {
        return rocrand_device::detail::uniform_distribution(v);
    }
};

template<>
struct sobol_uniform_distribution<double>
{
    __host__ __device__
    double operator()(const unsigned int v) const
    {
        return rocrand_device::detail::uniform_distribution_double(v);
    }
};

template<>
struct sobol_uniform_distribution<__half>
{
    __host__ __device__
    __half operator()(const unsigned int v) const
    {
        return rocrand_device::detail::uniform_distribution_half(static_cast<unsigned short>(v >> 16));
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_UNIFORM_H_
