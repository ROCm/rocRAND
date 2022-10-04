// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_RNG_THREEFRY2X64_20_H_
#define ROCRAND_RNG_THREEFRY2X64_20_H_

#include <algorithm>
#include <hip/hip_runtime.h>

#include <rocrand/rocrand.h>

#include "common.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"

namespace rocrand_host
{
namespace detail
{
struct threefry2x64_20_device_engine : public ::rocrand_device::threefry2x64_20_engine
{
    typedef ::rocrand_device::threefry2x64_20_engine base_type;
    typedef base_type::threefry2x64_20_state         state_type;

    __forceinline__ __device__ __host__ threefry2x64_20_device_engine() {}

    __forceinline__ __device__ __host__
        threefry2x64_20_device_engine(const unsigned long long seed,
                                      const unsigned long long subsequence,
                                      const unsigned long long offset)
        : base_type(seed, subsequence, offset)
    {}

    __forceinline__ __device__ __host__ ~threefry2x64_20_device_engine() {}

    __forceinline__ __device__ __host__ ulonglong2 next2_leap(unsigned int leap)
    {
        ulonglong2 ret = m_state.result;
        if(m_state.substate > 0)
        {
            const ulonglong2 next_counter = this->bump_counter(m_state.counter);
            const ulonglong2 next         = this->threefry_rounds(next_counter, m_state.key);
            ret                           = this->interleave(ret, next);
        }

        this->discard_state(leap);
        m_state.result = this->threefry_rounds(m_state.counter, m_state.key);
        return ret;
    }

    // m_state from base class
};

template<class T, class Distribution>
ROCRAND_KERNEL __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE) void generate_kernel(
    threefry2x64_20_device_engine engine, T* data, const size_t n, Distribution distribution)
{
    constexpr unsigned int input_width  = Distribution::input_width;
    constexpr unsigned int output_width = Distribution::output_width;

    static_assert(2 % input_width == 0 && input_width <= 2, "Incorrect input_width");
    constexpr unsigned int output_per_thread = 2 / input_width;
    constexpr unsigned int full_output_width = output_per_thread * output_width;

    using vec_type = aligned_vec_type<T, output_per_thread * output_width>;

    const unsigned int thread_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    const unsigned int stride    = hipGridDim_x * hipBlockDim_x;

    unsigned long long input[input_width];
    T                  output[output_per_thread][output_width];

    const uintptr_t uintptr = reinterpret_cast<uintptr_t>(data);
    const size_t    misalignment
        = (full_output_width - uintptr / sizeof(T) % full_output_width) % full_output_width;
    const unsigned int head_size = min(n, misalignment);
    const unsigned int tail_size = (n - head_size) % full_output_width;
    const size_t       vec_n     = (n - head_size) / full_output_width;

    const unsigned int engine_offset = 2 * thread_id + (thread_id == 0 ? 0 : head_size);
    engine.discard(engine_offset);

    // If data is not aligned by sizeof(vec_type)
    if(thread_id == 0 && head_size > 0)
    {
        for(unsigned int s = 0; s < output_per_thread; ++s)
        {
            if(s * output_width >= head_size)
            {
                break;
            }

            for(unsigned int i = 0; i < input_width; ++i)
            {
                input[i] = engine();
            }
            distribution(input, output[s]);

            for(unsigned int o = 0; o < output_width; ++o)
            {
                if(s * output_width + o < head_size)
                {
                    data[s * output_width + o] = output[s][o];
                }
            }
        }
    }

    // Save multiple values as one vec_type
    vec_type* vec_data = reinterpret_cast<vec_type*>(data + misalignment);
    size_t    index    = thread_id;
    while(index < vec_n)
    {
        const ulonglong2         v     = engine.next2_leap(stride);
        const unsigned long long vs[2] = {v.x, v.y};
        for(unsigned int s = 0; s < output_per_thread; s++)
        {
            for(unsigned int i = 0; i < input_width; i++)
            {
                input[i] = vs[s * input_width + i];
            }
            distribution(input, output[s]);
        }
        vec_data[index] = *reinterpret_cast<vec_type*>(output);
        // Next position
        index += stride;
    }

    // Check if we need to save tail.
    // Those numbers should be generated by the thread that would
    // save next vec_type.
    if(index == vec_n && tail_size > 0)
    {
        for(unsigned int s = 0; s < output_per_thread; ++s)
        {
            if(s * output_width >= tail_size)
            {
                break;
            }

            for(unsigned int i = 0; i < input_width; ++i)
            {
                input[i] = engine();
            }
            distribution(input, output[s]);

            for(unsigned int o = 0; o < output_width; ++o)
            {
                if(s * output_width + o < tail_size)
                {
                    data[n - tail_size + s * output_width + o] = output[s][o];
                }
            }
        }
    }
}

} // end namespace detail
} // end namespace rocrand_host

class rocrand_threefry2x64_20 : public rocrand_generator_type<ROCRAND_RNG_PSEUDO_THREEFRY2_64_20>
{
public:
    using base_type   = rocrand_generator_type<ROCRAND_RNG_PSEUDO_THREEFRY2_64_20>;
    using engine_type = ::rocrand_host::detail::threefry2x64_20_device_engine;

    rocrand_threefry2x64_20(unsigned long long seed   = 0,
                            unsigned long long offset = 0,
                            hipStream_t        stream = 0)
        : base_type(seed, offset, stream), m_engines_initialized(false)
    {}

    void reset()
    {
        m_engines_initialized = false;
    }

    /// Changes seed to \p seed and resets generator state.
    void set_seed(unsigned long long seed)
    {
        m_seed                = seed;
        m_engines_initialized = false;
    }

    void set_offset(unsigned long long offset)
    {
        m_offset              = offset;
        m_engines_initialized = false;
    }

    rocrand_status init()
    {
        if(m_engines_initialized)
            return ROCRAND_STATUS_SUCCESS;

        m_engine = engine_type{m_seed, 0, m_offset};

        m_engines_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T, unsigned long long>>
    rocrand_status generate(T* data, size_t data_size, Distribution distribution = Distribution())
    {
        rocrand_status status = init();
        if(status != ROCRAND_STATUS_SUCCESS)
            return status;

        hipLaunchKernelGGL(HIP_KERNEL_NAME(rocrand_host::detail::generate_kernel),
                           dim3(s_blocks),
                           dim3(s_threads),
                           0,
                           m_stream,
                           m_engine,
                           data,
                           data_size,
                           distribution);
        // Check kernel status
        if(hipGetLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        // Generating data_size values will use this many distributions
        const auto num_applied_generators = (data_size + Distribution::output_width - 1)
                                            / Distribution::output_width
                                            * Distribution::input_width;

        m_engine.discard(num_applied_generators);

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T* data, size_t data_size)
    {
        uniform_distribution<T, unsigned long long> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T* data, size_t data_size, T mean, T stddev)
    {
        normal_distribution<T, unsigned long long> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T* data, size_t data_size, T mean, T stddev)
    {
        log_normal_distribution<T, unsigned long long> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_poisson(T* data, size_t data_size, double lambda)
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
    bool        m_engines_initialized;
    engine_type m_engine;

    const static uint32_t s_threads = 256;
    const static uint32_t s_blocks  = 1024;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<> m_poisson;

    // m_seed from base_type
    // m_offset from base_type
};

#endif // ROCRAND_RNG_THREEFRY2X64_20_H_
