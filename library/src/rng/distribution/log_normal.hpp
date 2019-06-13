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

#include <math.h>
#include <hip/hip_runtime.h>

#include "common.hpp"
#include "device_distributions.hpp"

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
        float2 v = rocrand_device::detail::box_muller(x, y);
        v.x = expf(mean + (stddev * v.x));
        v.y = expf(mean + (stddev * v.y));
        return v;
    }

    __forceinline__ __host__ __device__
    float4 operator()(const uint4 x)
    {
        float2 v = rocrand_device::detail::box_muller(x.x, x.y);
        float2 w = rocrand_device::detail::box_muller(x.z, x.w);
        return float4 {
            expf(mean + (stddev * v.x)),
            expf(mean + (stddev * v.y)),
            expf(mean + (stddev * w.x)),
            expf(mean + (stddev * w.y))
        };
    }

    __forceinline__ __host__ __device__
    float operator()(unsigned int x)
    {
        float v = rocrand_device::detail::normal_distribution(x);
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
        double2 v = rocrand_device::detail::box_muller_double(x);
        v.x = exp(mean + (stddev * v.x));
        v.y = exp(mean + (stddev * v.y));
        return v;
    }

    __forceinline__ __host__ __device__
    double operator()(unsigned int x)
    {
        double v = rocrand_device::detail::normal_distribution_double(x);
        v = exp(mean + (stddev * v));
        return v;
    }
};

template<>
struct log_normal_distribution<__half>
{
    const __half mean;
    const __half stddev;

    __host__ __device__
    log_normal_distribution<__half>(const __half mean, const __half stddev) :
                                    mean(mean), stddev(stddev) {}

    __forceinline__ __device__
    rocrand_half2 operator()(const unsigned int x)
    {
        rocrand_half2 v =
            box_muller_half(static_cast<short>(x), static_cast<short>(x >> 16));
        #if defined(__HIP_PLATFORM_HCC__) || ((__CUDA_ARCH__ >= 530) && defined(__HIP_PLATFORM_NVCC__))
        return rocrand_half2 {
            hexp(__hadd(mean, __hmul(stddev, v.x))),
            hexp(__hadd(mean, __hmul(stddev, v.y)))
        };
        #else
        return rocrand_half2 {
            __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(v.x)))),
            __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(v.y))))
        };
        #endif
    }

    __forceinline__ __device__
    rocrand_half4 operator()(const unsigned int x, const unsigned int y)
    {
        rocrand_half2 v =
            box_muller_half(static_cast<short>(x), static_cast<short>(x >> 16));
        rocrand_half2 w =
            box_muller_half(static_cast<short>(y), static_cast<short>(y >> 16));
        #if defined(__HIP_PLATFORM_HCC__) || ((__CUDA_ARCH__ >= 530) && defined(__HIP_PLATFORM_NVCC__))
        return rocrand_half4 {
            rocrand_half2 {
                hexp(__hadd(mean, __hmul(stddev, v.x))),
                hexp(__hadd(mean, __hmul(stddev, v.y)))
            },
            rocrand_half2 {
                hexp(__hadd(mean, __hmul(stddev, w.x))),
                hexp(__hadd(mean, __hmul(stddev, w.y)))
            }
        };
        #else
        return rocrand_half4 {
            rocrand_half2 {
                __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(v.x)))),
                __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(v.y))))
            },
            rocrand_half2 {
                __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(w.x)))),
                __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(w.y))))
            }
        };
        #endif
    }

    __forceinline__ __device__
    rocrand_half8 operator()(const uint4 x)
    {
        rocrand_half2 t =
            box_muller_half(static_cast<short>(x.x), static_cast<short>(x.x >> 16));
        rocrand_half2 u =
            box_muller_half(static_cast<short>(x.y), static_cast<short>(x.y >> 16));
        rocrand_half2 v =
            box_muller_half(static_cast<short>(x.z), static_cast<short>(x.z >> 16));
        rocrand_half2 w =
            box_muller_half(static_cast<short>(x.w), static_cast<short>(x.w >> 16));
        #if defined(__HIP_PLATFORM_HCC__) || ((__CUDA_ARCH__ >= 530) && defined(__HIP_PLATFORM_NVCC__))
        return rocrand_half8 {
            rocrand_half4 {
                rocrand_half2 {
                    hexp(__hadd(mean, __hmul(stddev, t.x))),
                    hexp(__hadd(mean, __hmul(stddev, t.y)))
                },
                rocrand_half2 {
                    hexp(__hadd(mean, __hmul(stddev, u.x))),
                    hexp(__hadd(mean, __hmul(stddev, u.y)))
                }
            },
            rocrand_half4 {
                rocrand_half2 {
                    hexp(__hadd(mean, __hmul(stddev, v.x))),
                    hexp(__hadd(mean, __hmul(stddev, v.y)))
                },
                rocrand_half2 {
                    hexp(__hadd(mean, __hmul(stddev, w.x))),
                    hexp(__hadd(mean, __hmul(stddev, w.y)))
                }
            }
        };
        #else
        return rocrand_half8 {
            rocrand_half4 {
                rocrand_half2 {
                    __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(t.x)))),
                    __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(t.y))))
                },
                rocrand_half2 {
                    __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(u.x)))),
                    __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(u.y))))
                }
            },
            rocrand_half4 {
                rocrand_half2 {
                    __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(v.x)))),
                    __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(v.y))))
                },
                rocrand_half2 {
                    __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(w.x)))),
                    __float2half(expf(__half2float(mean) + (__half2float(stddev) * __half2float(w.y))))
                }
            }
        };
        #endif
    }
};

#endif // ROCRAND_RNG_DISTRIBUTION_LOG_NORMAL_H_
