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

#ifndef ROCRAND_RNG_MRG32K3A_H_
#define ROCRAND_RNG_MRG32K3A_H_

#include <algorithm>
#include <hip/hip_runtime.h>

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_mrg32k3a_precomputed.h>

#include "generator_type.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"

// MRG32K3A constants
#ifndef ROCRAND_MRG32K3A_NORM_DOUBLE
#define ROCRAND_MRG32K3A_NORM_DOUBLE (2.3283065498378288e-10) // 1/ROCRAND_MRG32K3A_M1
#endif
#ifndef ROCRAND_MRG32K3A_UINT_NORM
#define ROCRAND_MRG32K3A_UINT_NORM (1.000000048661606966) // ROCRAND_MRG32K3A_POW32/ROCRAND_MRG32K3A_M1
#endif

namespace rocrand_host {
namespace detail {

    typedef ::rocrand_device::mrg32k3a_engine mrg32k3a_device_engine;

    __global__
    void init_engines_kernel(mrg32k3a_device_engine * engines,
                             unsigned long long seed,
                             unsigned long long offset)
    {
        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        engines[engine_id] = mrg32k3a_device_engine(seed, engine_id, offset);
    }

    template<class Type, class Distribution>
    __global__
    typename std::enable_if<std::is_same<Type, unsigned int>::value
                            || std::is_same<Type, float>::value
                            || std::is_same<Type, double>::value>::type
    generate_kernel(mrg32k3a_device_engine * engines,
                    Type * data, const size_t n,
                    const Distribution distribution)
    {
        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = engine_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        mrg32k3a_device_engine engine = engines[engine_id];

        while(index < n)
        {
            data[index] = distribution(engine());
            // Next position
            index += stride;
        }

        // Save engine with its state
        engines[engine_id] = engine;
    }

    template<class Type, class Distribution>
    __global__
    typename std::enable_if<std::is_same<Type, unsigned char>::value>::type
    generate_kernel(mrg32k3a_device_engine * engines,
                    Type * data, const size_t n,
                    const Distribution distribution)
    {
        typedef decltype(distribution(engines->next())) Type4;

        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = engine_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        mrg32k3a_device_engine engine = engines[engine_id];

        Type4 * data4 = (Type4 *) data;
        while(index < (n / 4))
        {
            data4[index] = distribution(engine());
            // Next position
            index += stride;
        }

        auto tail_size = n & 3;
        if((index == n/4) && tail_size > 0)
        {
            Type4 result = distribution(engine());
            // Save the tail
            data[n - tail_size] = result.x;
            if(tail_size > 1) data[n - tail_size + 1] = result.y;
            if(tail_size > 2) data[n - tail_size + 2] = result.z;
        }

        // Save engine with its state
        engines[engine_id] = engine;
    }

    template<class Type, class Distribution>
    __global__
    typename std::enable_if<std::is_same<Type, unsigned short>::value
                            || std::is_same<Type, __half>::value>::type
    generate_kernel(mrg32k3a_device_engine * engines,
                    Type * data, const size_t n,
                    const Distribution distribution)
    {
        typedef decltype(distribution(engines->next())) Type2;

        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = engine_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        mrg32k3a_device_engine engine = engines[engine_id];

        Type2 * data2 = (Type2 *) data;
        while(index < (n / 2))
        {
            data2[index] = distribution(engine());
            // Next position
            index += stride;
        }

        // First work-item saves the tail when n is not a multiple of 2
        if(engine_id == 0 && (n & 1) > 0)
        {
            Type2 result = distribution(engine());
            // Save the tail
            data[n - 1] = result.x;
        }

        // Save engine with its state
        engines[engine_id] = engine;
    }

    template<class RealType, class Distribution>
    __global__
    typename std::enable_if<!std::is_same<RealType, __half>::value>::type
    generate_normal_kernel(mrg32k3a_device_engine * engines,
                           RealType * data, const size_t n,
                           Distribution distribution)
    {
        typedef decltype(distribution(engines->next(), engines->next())) RealType2;

        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = engine_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        mrg32k3a_device_engine engine = engines[engine_id];

        RealType2 * data2 = (RealType2 *) data;
        while(index < (n / 2))
        {
            data2[index] = distribution(engine(), engine());
            // Next position
            index += stride;
        }

        // First work-item saves the tail when n is not a multiple of 2
        if(engine_id == 0 && (n & 1) > 0)
        {
            RealType2 result = distribution(engine(), engine());
            // Save the tail
            data[n - 1] = result.x;
        }

        // Save engine with its state
        engines[engine_id] = engine;
    }

    template<class RealType, class Distribution>
    __global__
    typename std::enable_if<std::is_same<RealType, __half>::value>::type
    generate_normal_kernel(mrg32k3a_device_engine * engines,
                           RealType * data, const size_t n,
                           Distribution distribution)
    {
        typedef decltype(distribution(engines->next(), engines->next())) RealType4;

        const unsigned int engine_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = engine_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load device engine
        mrg32k3a_device_engine engine = engines[engine_id];

        RealType4 * data4 = (RealType4 *) data;
        while(index < (n / 4))
        {
            data4[index] = distribution(engine(), engine());
            // Next position
            index += stride;
        }

        auto tail_size = n & 3;
        if((index == n / 4) && tail_size > 0)
        {
            RealType4 result = distribution(engine(), engine());
            // Save the tail
            data[n - tail_size] = result.x;
            if(tail_size > 1) data[n - tail_size + 1] = result.y;
            if(tail_size > 2) data[n - tail_size + 2] = result.z;
        }

        // Save engine with its state
        engines[engine_id] = engine;
    }

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

    template<>
    struct mrg_uniform_distribution<unsigned char>
    {
        __forceinline__ __host__ __device__
        uchar4 operator()(const unsigned int v) const
        {
            unsigned int w = static_cast<unsigned int>(v * ROCRAND_MRG32K3A_UINT_NORM);
            return make_uchar4(
                (unsigned char)(w),
                (unsigned char)(w >> 8),
                (unsigned char)(w >> 16),
                (unsigned char)(w >> 24)
            );
        }
    };

    template<>
    struct mrg_uniform_distribution<unsigned short>
    {
        __forceinline__ __host__ __device__
        ushort2 operator()(const unsigned int v) const
        {
            unsigned int w = static_cast<unsigned int>(v * ROCRAND_MRG32K3A_UINT_NORM);
            return make_ushort2(
                (unsigned short)(w),
                (unsigned short)(w >> 16)
            );
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
            return rocrand_device::detail::mrg_uniform_distribution(v);
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
            return rocrand_device::detail::mrg_uniform_distribution_double(v);
        }
    };

    template<>
    struct mrg_uniform_distribution<__half>
    {
        __forceinline__ __host__ __device__
        rocrand_half2 operator()(const unsigned int v) const
        {
            unsigned int w = static_cast<unsigned int>(v * ROCRAND_MRG32K3A_UINT_NORM);
            return rocrand_half2 {
                uniform_distribution_half(static_cast<short>(w)),
                uniform_distribution_half(static_cast<short>(w >> 16))
            };
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

    template<>
    struct mrg_normal_distribution<__half>
    {
        const __half mean;
        const __half stddev;

        __host__ __device__
        mrg_normal_distribution<__half>(__half mean = 0.0f, __half stddev = 1.0f) :
                                        mean(mean), stddev(stddev) {}

        __forceinline__ __host__ __device__
        rocrand_half4 operator()(const unsigned int x, const unsigned int y)
        {
            unsigned int a = static_cast<unsigned int>(x * ROCRAND_MRG32K3A_UINT_NORM);
            unsigned int b = static_cast<unsigned int>(y * ROCRAND_MRG32K3A_UINT_NORM);
            rocrand_half4 m = rocrand_half4 (
                rocrand_half2 {
                    uniform_distribution_half(static_cast<short>(a)),
                    uniform_distribution_half(static_cast<short>(a >> 16))
                },
                rocrand_half2 {
                    uniform_distribution_half(static_cast<short>(b)),
                    uniform_distribution_half(static_cast<short>(b >> 16))
                }
            );
            rocrand_half2 v = mrg_box_muller_half(m.x, m.y);
            rocrand_half2 w = mrg_box_muller_half(m.z, m.w);
            #if defined(__HIP_PLATFORM_HCC__) || ((__CUDA_ARCH__ >= 530) && defined(__HIP_PLATFORM_NVCC__))
            return rocrand_half4 {
                rocrand_half2 {
                    __hadd(mean, __hmul(v.x, stddev)),
                    __hadd(mean, __hmul(v.y, stddev))
                },
                rocrand_half2 {
                    __hadd(mean, __hmul(w.x, stddev)),
                    __hadd(mean, __hmul(w.y, stddev))
                }
            };
            #else
            return rocrand_half4 {
                rocrand_half2 {
                    __float2half(__half2float(mean) + (__half2float(stddev) * __half2float(v.x))),
                    __float2half(__half2float(mean) + (__half2float(stddev) * __half2float(v.y)))
                },
                rocrand_half2 {
                    __float2half(__half2float(mean) + (__half2float(stddev) * __half2float(w.x))),
                    __float2half(__half2float(mean) + (__half2float(stddev) * __half2float(w.y)))
                }
            };
            #endif
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
        float2 operator()(const unsigned int x, const unsigned int y)
        {
            float2 v = rocrand_device::detail::mrg_normal_distribution2(x, y);
            v.x = expf(mean + (stddev * v.x));
            v.y = expf(mean + (stddev * v.y));
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
        double2 operator()(const unsigned int x, const unsigned int y)
        {
            double2 v = rocrand_device::detail::mrg_normal_distribution_double2(x, y);
            v.x = exp(mean + (stddev * v.x));
            v.y = exp(mean + (stddev * v.y));
            return v;
        }
    };

    template<>
    struct mrg_log_normal_distribution<__half>
    {
        const __half mean;
        const __half stddev;

        __host__ __device__
        mrg_log_normal_distribution<__half>(__half mean = 0.0f, __half stddev = 1.0f) :
                                            mean(mean), stddev(stddev) {}

        __forceinline__ __host__ __device__
        rocrand_half4 operator()(const unsigned int x, const unsigned int y)
        {
            unsigned int a = static_cast<unsigned int>(x * ROCRAND_MRG32K3A_UINT_NORM);
            unsigned int b = static_cast<unsigned int>(y * ROCRAND_MRG32K3A_UINT_NORM);
            rocrand_half4 m = rocrand_half4 (
                rocrand_half2 {
                    uniform_distribution_half(static_cast<short>(a)),
                    uniform_distribution_half(static_cast<short>(a >> 16))
                },
                rocrand_half2 {
                    uniform_distribution_half(static_cast<short>(b)),
                    uniform_distribution_half(static_cast<short>(b >> 16))
                }
            );
            rocrand_half2 v = mrg_box_muller_half(m.x, m.y);
            rocrand_half2 w = mrg_box_muller_half(m.z, m.w);
            #if defined(__HIP_PLATFORM_HCC__) || ((__CUDA_ARCH__ >= 530) && defined(__HIP_PLATFORM_NVCC__))
            return rocrand_half4 {
                rocrand_half2 {
                    hexp(__hadd(mean, __hmul(stddev, (v.x)))),
                    hexp(__hadd(mean, __hmul(stddev, (v.y))))
                },
                rocrand_half2 {
                    hexp(__hadd(mean, __hmul(stddev, (w.x)))),
                    hexp(__hadd(mean, __hmul(stddev, (w.y))))
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
    };

} // end namespace detail
} // end namespace rocrand_host

class rocrand_mrg32k3a : public rocrand_generator_type<ROCRAND_RNG_PSEUDO_MRG32K3A>
{
public:
    using base_type = rocrand_generator_type<ROCRAND_RNG_PSEUDO_MRG32K3A>;
    using engine_type = ::rocrand_host::detail::mrg32k3a_device_engine;

    rocrand_mrg32k3a(unsigned long long seed = 12345,
                     unsigned long long offset = 0,
                     hipStream_t stream = 0)
        : base_type(seed, offset, stream),
          m_engines_initialized(false), m_engines(NULL), m_engines_size(s_threads * s_blocks)
    {
        // Allocate device random number engines
        auto error = hipMalloc(&m_engines, sizeof(engine_type) * m_engines_size);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
        if(m_seed == 0)
        {
            m_seed = ROCRAND_MRG32K3A_DEFAULT_SEED;
        }
    }

    ~rocrand_mrg32k3a()
    {
        hipFree(m_engines);
    }

    void reset()
    {
        m_engines_initialized = false;
    }

    /// Changes seed to \p seed and resets generator state.
    ///
    /// New seed value should not be zero. If \p seed_value is equal
    /// zero, value \p ROCRAND_MRG32K3A_DEFAULT_SEED is used instead.
    void set_seed(unsigned long long seed)
    {
        if(seed == 0)
        {
            seed = ROCRAND_MRG32K3A_DEFAULT_SEED;
        }
        m_seed = seed;
        m_engines_initialized = false;
    }

    void set_offset(unsigned long long offset)
    {
        m_offset = offset;
        m_engines_initialized = false;
    }

    rocrand_status init()
    {
        if (m_engines_initialized)
            return ROCRAND_STATUS_SUCCESS;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::init_engines_kernel),
            dim3(s_blocks), dim3(s_threads), 0, m_stream,
            m_engines, m_seed, m_offset
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_engines_initialized = true;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = rocrand_host::detail::mrg_uniform_distribution<T> >
    rocrand_status generate(T * data, size_t data_size,
                            const Distribution& distribution = Distribution())
    {
        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_kernel),
            dim3(s_blocks), dim3(s_threads), 0, m_stream,
            m_engines, data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T * data, size_t data_size)
    {
        rocrand_host::detail::mrg_uniform_distribution<T> udistribution;
        return generate(data, data_size, udistribution);
    }

    template<class T>
    rocrand_status generate_normal(T * data, size_t data_size, T mean, T stddev)
    {
        // data_size must be even
        // data must be aligned to 2 * sizeof(T) bytes
        if(data_size%2 != 0 || ((uintptr_t)(data)%(2*sizeof(T))) != 0)
        {
            return ROCRAND_STATUS_LENGTH_NOT_MULTIPLE;
        }

        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        rocrand_host::detail::mrg_normal_distribution<T> distribution(mean, stddev);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_normal_kernel),
            dim3(s_blocks), dim3(s_threads), 0, m_stream,
            m_engines, data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_log_normal(T * data, size_t data_size, T mean, T stddev)
    {
        // data_size must be even
        // data must be aligned to 2 * sizeof(T) bytes
        if(data_size%2 != 0 || ((uintptr_t)(data)%(2*sizeof(T))) != 0)
        {
            return ROCRAND_STATUS_LENGTH_NOT_MULTIPLE;
        }

        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        rocrand_host::detail::mrg_log_normal_distribution<T> distribution(mean, stddev);

        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(rocrand_host::detail::generate_normal_kernel),
            dim3(s_blocks), dim3(s_threads), 0, m_stream,
            m_engines, data, data_size, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status generate_poisson(unsigned int * data, size_t data_size, double lambda)
    {
        try
        {
            m_poisson.set_lambda(lambda);
        }
        catch(rocrand_status status)
        {
            return status;
        }
        return generate(data, data_size, m_poisson.dis);
    }

private:
    bool m_engines_initialized;
    engine_type * m_engines;
    size_t m_engines_size;
    #ifdef __HIP_PLATFORM_NVCC__
    static const uint32_t s_threads = 128;
    static const uint32_t s_blocks = 128;
    #else
    static const uint32_t s_threads = 256;
    static const uint32_t s_blocks = 512;
    #endif

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<> m_poisson;

    // m_seed from base_type
    // m_offset from base_type
};

#endif // ROCRAND_RNG_MRG32K3A_H_
