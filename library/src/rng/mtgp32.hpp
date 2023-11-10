// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

/*
 * Copyright (c) 2009, 2010 Mutsuo Saito, Makoto Matsumoto and Hiroshima
 * University.  All rights reserved.
 * Copyright (c) 2011 Mutsuo Saito, Makoto Matsumoto, Hiroshima
 * University and University of Tokyo.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 *       copyright notice, this list of conditions and the following
 *       disclaimer in the documentation and/or other materials provided
 *       with the distribution.
 *     * Neither the name of the Hiroshima University nor the names of
 *       its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written
 *       permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef ROCRAND_RNG_MTGP32_H_
#define ROCRAND_RNG_MTGP32_H_

#include <algorithm>
#include <hip/hip_runtime.h>

// common.hpp MUST be included prior to the device engines
// to correctly define FQUALIFIERS
#include "common.hpp"

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_mtgp32_11213.h>

#include "config/config_defaults.hpp"
#include "config_types.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"

namespace rocrand_host::detail
{

typedef ::rocrand_device::mtgp32_engine mtgp32_device_engine;

template<class ConfigProvider, bool IsDynamic, class T, class Distribution>
ROCRAND_KERNEL
    __launch_bounds__((get_block_size<ConfigProvider, T>(IsDynamic))) void generate_kernel(
        mtgp32_device_engine* engines, T* data, const size_t n, Distribution distribution)
{
    constexpr unsigned int BlockSize    = get_block_size<ConfigProvider, T>(IsDynamic);
    constexpr unsigned int input_width  = Distribution::input_width;
    constexpr unsigned int output_width = Distribution::output_width;

    using vec_type = aligned_vec_type<T, output_width>;

    const unsigned int engine_id = blockIdx.x;
    const unsigned int stride    = gridDim.x * BlockSize;
    size_t             index     = blockIdx.x * BlockSize + threadIdx.x;

    // Load device engine
    __shared__ mtgp32_device_engine engine;
    engine.copy(&engines[engine_id]);

    unsigned int input[input_width];
    T            output[output_width];

    const uintptr_t uintptr   = reinterpret_cast<uintptr_t>(data);
    const size_t misalignment = (output_width - uintptr / sizeof(T) % output_width) % output_width;
    const unsigned int head_size = min(n, misalignment);
    const unsigned int tail_size = (n - head_size) % output_width;
    const size_t       vec_n     = (n - head_size) / output_width;

    // vec_n rounded up and down to the nearest multiple of BlockSize
    const size_t remainder_value = vec_n % BlockSize;
    const size_t vec_n_down      = vec_n - remainder_value;
    const size_t vec_n_up        = remainder_value == 0 ? vec_n_down : (vec_n_down + BlockSize);

    vec_type* vec_data = reinterpret_cast<vec_type*>(data + misalignment);
    while(index < vec_n_down)
    {
        for(unsigned int i = 0; i < input_width; i++)
        {
            input[i] = engine();
        }
        distribution(input, output);

        vec_data[index] = *reinterpret_cast<vec_type*>(output);
        // Next position
        index += stride;
    }
    if(index < vec_n_up)
    {
        for(unsigned int i = 0; i < input_width; i++)
        {
            input[i] = engine();
        }
        distribution(input, output);

        // All threads generate (hence call __syncthreads) but not all write
        if(index < vec_n)
        {
            vec_data[index] = *reinterpret_cast<vec_type*>(output);
        }
        // Next position
        index += stride;
    }

    // Check if we need to save head and tail.
    if(output_width > 1 && (head_size > 0 || tail_size > 0))
    {
        for(unsigned int i = 0; i < input_width; i++)
        {
            input[i] = engine();
        }
        distribution(input, output);

        // If data is not aligned by sizeof(vec_type)
        if(index == vec_n_up)
        {
            for(unsigned int o = 0; o < output_width; o++)
            {
                if(o < head_size)
                {
                    data[o] = output[o];
                }
            }
        }

        if(index == vec_n_up + 1)
        {
            for(unsigned int o = 0; o < output_width; o++)
            {
                if(o < tail_size)
                {
                    data[n - tail_size + o] = output[o];
                }
            }
        }
    }

    // Save engine with its state
    engines[engine_id].copy(&engine);
}

} // end namespace rocrand_host::detail

template<class ConfigProvider>
class rocrand_mtgp32_template : public rocrand_generator_impl_base
{
public:
    using base_type   = rocrand_generator_impl_base;
    using engine_type = ::rocrand_host::detail::mtgp32_device_engine;

    rocrand_mtgp32_template(unsigned long long seed   = 0,
                            unsigned long long offset = 0,
                            rocrand_ordering   order  = ROCRAND_ORDERING_PSEUDO_DEFAULT,
                            hipStream_t        stream = 0)
        : base_type(order, offset, stream), m_seed(seed)
    {}

    rocrand_mtgp32_template(const rocrand_mtgp32_template&) = delete;

    rocrand_mtgp32_template(rocrand_mtgp32_template&& other)
        : base_type(other)
        , m_engines_initialized(other.m_engines_initialized)
        , m_engines(other.m_engines)
        , m_engines_size(other.m_engines_size)
        , m_seed(other.m_seed)
        , m_poisson(std::move(other.m_poisson))
    {
        other.m_engines_initialized = false;
        other.m_engines             = nullptr;
    }

    rocrand_mtgp32_template& operator=(const rocrand_mtgp32_template&&) = delete;

    rocrand_mtgp32_template& operator=(rocrand_mtgp32_template&& other)
    {
        *static_cast<base_type*>(this) = other;
        m_engines_initialized          = other.m_engines_initialized;
        m_engines                      = other.m_engines;
        m_engines_size                 = other.m_engines_size;
        m_seed                         = other.m_seed;
        m_poisson                      = std::move(other.m_poisson);

        other.m_engines_initialized = false;
        other.m_engines             = nullptr;

        return *this;
    }

    ~rocrand_mtgp32_template()
    {
        if(m_engines != nullptr)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(m_engines));
        }
    }

    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_MTGP32;
    }

    void reset() override final
    {
        m_engines_initialized = false;
    }

    /// Changes seed to \p seed and resets generator state.
    void set_seed(unsigned long long seed)
    {
        m_seed = seed;
        reset();
    }

    unsigned long long get_seed() const
    {
        return m_seed;
    }

    rocrand_status set_offset(unsigned long long offset)
    {
        // Can't set offset for MTGP32
        (void)offset;
        return ROCRAND_STATUS_TYPE_ERROR;
    }

    rocrand_status set_order(rocrand_ordering order)
    {
        if(!rocrand_host::detail::is_ordering_pseudo(order))
        {
            return ROCRAND_STATUS_OUT_OF_RANGE;
        }
        m_order = order;
        reset();
        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status init()
    {
        if (m_engines_initialized)
        {
            return ROCRAND_STATUS_SUCCESS;
        }

        rocrand_host::detail::generator_config config;
        // Assuming that the config is the same for every type.
        hipError_t error
            = ConfigProvider{}.template host_config<unsigned int>(m_stream, m_order, config);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        m_engines_size = config.blocks;

        if (m_engines_size > mtgpdc_params_11213_num)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        error
            = hipMalloc(reinterpret_cast<void**>(&m_engines), sizeof(engine_type) * m_engines_size);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        rocrand_status status = rocrand_make_state_mtgp32(m_engines,
                                                          mtgp32dc_params_fast_11213,
                                                          m_engines_size,
                                                          m_seed);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        m_engines_initialized = true;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T> >
    rocrand_status generate(T * data, size_t data_size,
                            Distribution distribution = Distribution())
    {
        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        rocrand_host::detail::generator_config config;
        const hipError_t                       error
            = ConfigProvider{}.template host_config<T>(m_stream, m_order, config);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        ROCRAND_LAUNCH_KERNEL_FOR_ORDERING(T,
                                           m_order,
                                           rocrand_host::detail::generate_kernel,
                                           dim3(config.blocks),
                                           dim3(config.threads),
                                           0,
                                           m_stream,
                                           m_engines,
                                           data,
                                           data_size,
                                           distribution);

        // Check kernel status
        if(hipGetLastError() != hipSuccess)
        {
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }

        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status generate(unsigned long long* data, size_t data_size)
    {
        // Cannot generate 64-bit values with this generator.
        (void)data;
        (void)data_size;
        return ROCRAND_STATUS_TYPE_ERROR;
    }

    template<typename Distribution>
    rocrand_status generate(unsigned long long* data, size_t data_size, Distribution distribution)
    {
        // Cannot generate 64-bit values with this generator.
        (void)data;
        (void)data_size;
        (void)distribution;
        return ROCRAND_STATUS_TYPE_ERROR;
    }

    template<class T>
    rocrand_status generate_uniform(T * data, size_t data_size)
    {
        uniform_distribution<T> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T * data, size_t data_size, T mean, T stddev)
    {
        normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T * data, size_t data_size, T mean, T stddev)
    {
        log_normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
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
    bool         m_engines_initialized = false;
    engine_type* m_engines             = nullptr;
    unsigned int m_engines_size        = false;

    unsigned long long m_seed;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<> m_poisson;

    // m_seed from base_type
    // m_offset from base_type
};

using rocrand_mtgp32 = rocrand_mtgp32_template<
    rocrand_host::detail::default_config_provider<ROCRAND_RNG_PSEUDO_MTGP32>>;

#endif // ROCRAND_RNG_MTGP32_H_
