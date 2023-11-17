// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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

// Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//  1. Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//  2. Redistributions in binary form must reproduce the above copyright
//     notice, this list of conditions and the following disclaimer in the
//     documentation and/or other materials provided with the distribution.
//
//  3. The names of its contributors may not be used to endorse or promote
//     products derived from this software without specific prior written
//     permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef ROCRAND_RNG_MT19937_H_
#define ROCRAND_RNG_MT19937_H_

#include "common.hpp"
#include "config_types.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"
#include "mt19937_octo_engine.hpp"

#include "config/config_defaults.hpp"
#include "config_types.hpp"

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_mt19937_precomputed.h>

#include <hip/hip_runtime.h>

#include <utility>

namespace rocrand_host::detail
{

/// Computes i % n, i must be in range [0, 2 * n)
MT_FQUALIFIERS unsigned int wrap_n(unsigned int i)
{
    return i - (i < mt19937_constants::n ? 0 : mt19937_constants::n);
}

// Config is not actually used for kernel launch here, but is needed to check the number of generators
// As this kernel is not dependent on any type just use void for the config, as mt19937 is not tuned for types independently, so all configs are the same for different types.
template<unsigned int jump_ahead_thread_count, class ConfigProvider, bool IsDynamic>
ROCRAND_KERNEL __launch_bounds__(jump_ahead_thread_count) void jump_ahead_kernel(
    unsigned int* __restrict__ engines,
    unsigned long long seed,
    const unsigned int* __restrict__ jump)
{
    constexpr generator_config config = ConfigProvider::template device_config<void>(IsDynamic);
    constexpr unsigned int     GeneratorCount
        = config.threads * config.blocks / mt19937_octo_engine::threads_per_generator;
    static_assert(GeneratorCount <= mt19937_jumps_radix * mt19937_jumps_radix
                      && mt19937_jumps_radixes == 2,
                  "Not enough rocrand_h_mt19937_jump values to initialize all generators");

    constexpr unsigned int block_size       = jump_ahead_thread_count;
    constexpr unsigned int items_per_thread = (mt19937_constants::n + block_size - 1) / block_size;
    constexpr unsigned int tail_n = mt19937_constants::n - (items_per_thread - 1) * block_size;

    __shared__ unsigned int temp[mt19937_constants::n];
    unsigned int            state[items_per_thread];

    // Initialize state 0 (engine_id = 0) used as a base for all engines.
    // It uses a recurrence relation so one thread calculates all n values.
    if(threadIdx.x == 0)
    {
        const unsigned int seedu = (seed >> 32) ^ seed;
        temp[0]                  = seedu;
        for(unsigned int i = 1; i < mt19937_constants::n; i++)
        {
            temp[i] = 1812433253 * (temp[i - 1] ^ (temp[i - 1] >> 30)) + i;
        }
    }
    __syncthreads();

    for(unsigned int i = 0; i < items_per_thread; i++)
    {
        if(i < items_per_thread - 1 || threadIdx.x < tail_n) // Check only for the last iteration
        {
            state[i] = temp[i * block_size + threadIdx.x];
        }
    }
    __syncthreads();

    const unsigned int engine_id = blockIdx.x;

    // Jump ahead by engine_id * 2 ^ 1000 using precomputed polynomials for jumps of
    // i * 2 ^ 1000 and mt19937_jumps_radix * i * 2 ^ 1000 values
    // where i is in range [1; mt19937_jumps_radix).
    unsigned int e = engine_id;
    for(unsigned int r = 0; r < mt19937_jumps_radixes; r++)
    {
        const unsigned int radix = e % mt19937_jumps_radix;
        e /= mt19937_jumps_radix;
        if(radix == 0)
        {
            continue;
        }

        // Compute jumping ahead with standard Horner method

        unsigned int ptr = 0;
        for(unsigned int i = threadIdx.x; i < mt19937_constants::n; i += block_size)
        {
            temp[i] = 0;
        }
        __syncthreads();

        const unsigned int* pf
            = jump + (r * (mt19937_jumps_radix - 1) + radix - 1) * mt19937_p_size;
        for(int pfi = mt19937_constants::mexp - 1; pfi >= 0; pfi--)
        {
            // Generate next state
            if(threadIdx.x == 0)
            {
                unsigned int t0 = temp[ptr];
                unsigned int t1 = temp[wrap_n(ptr + 1)];
                unsigned int tm = temp[wrap_n(ptr + mt19937_constants::m)];
                unsigned int y
                    = (t0 & mt19937_constants::upper_mask) | (t1 & mt19937_constants::lower_mask);
                temp[ptr] = tm ^ (y >> 1) ^ ((y & 0x1U) ? mt19937_constants::matrix_a : 0);
            }
            __syncthreads();
            ptr = wrap_n(ptr + 1);

            if((pf[pfi / 32] >> (pfi % 32)) & 1)
            {
                // Add state to temp
                for(unsigned int i = 0; i < items_per_thread; i++)
                {
                    if(i < items_per_thread - 1 || threadIdx.x < tail_n)
                    {
                        temp[wrap_n(ptr + i * block_size + threadIdx.x)] ^= state[i];
                    }
                }
                __syncthreads();
            }
        }

        // Jump of the next power of 2 will be applied to the current state
        for(unsigned int i = 0; i < items_per_thread; i++)
        {
            if(i < items_per_thread - 1 || threadIdx.x < tail_n)
            {
                state[i] = temp[wrap_n(ptr + i * block_size + threadIdx.x)];
            }
        }
        __syncthreads();
    }

    // Save state
    for(unsigned int i = 0; i < items_per_thread; i++)
    {
        if(i < items_per_thread - 1 || threadIdx.x < tail_n)
        {
            engines[engine_id * mt19937_constants::n + i * block_size + threadIdx.x] = state[i];
        }
    }
}

// This kernel is not explicitly tuned, but uses the same configs as the generate-kernels.
// As this kernel is not dependent on any type just use void for the config, as mt19937 is not tuned for types independently, so all configs are the same for different types.
template<class ConfigProvider, bool IsDynamic>
ROCRAND_KERNEL
    __launch_bounds__((get_block_size<ConfigProvider, void>(IsDynamic))) void init_engines_kernel(
        unsigned int* __restrict__ octo_engines, const unsigned int* __restrict__ engines)
{
    constexpr generator_config config     = ConfigProvider::template device_config<void>(IsDynamic);
    constexpr unsigned int     block_size = config.threads;
    constexpr unsigned int     grid_size  = config.blocks;
    constexpr unsigned int     stride     = block_size * grid_size;
    constexpr unsigned int     threads_per_generator = mt19937_octo_engine::threads_per_generator;
    // All threads of a generator must be in the same block
    static_assert(block_size % threads_per_generator == 0,
                  "All eight threads of the generator must be in the same block");

    const unsigned int thread_id = blockIdx.x * block_size + threadIdx.x;
    // every eight octo engines gather from the same engine
    mt19937_octo_engine_accessor<stride> accessor(octo_engines);
    mt19937_octo_engine engine;
    engine.gather(
        &engines[thread_id / mt19937_octo_engine::threads_per_generator * mt19937_constants::n]);
    accessor.save(thread_id, engine);
}

template<class ConfigProvider, bool IsDynamic, class T, class VecT, class Distribution>
ROCRAND_KERNEL __launch_bounds__((get_block_size<ConfigProvider, T>(
    IsDynamic))) void generate_short_kernel(unsigned int* __restrict__ engines,
                                            const unsigned int start_input,
                                            T* __restrict__ data,
                                            const size_t size,
                                            VecT* __restrict__ vec_data,
                                            const size_t       vec_size,
                                            const unsigned int head_size,
                                            const unsigned int tail_size,
                                            Distribution       distribution)
{
    constexpr generator_config config     = ConfigProvider::template device_config<T>(IsDynamic);
    constexpr unsigned int     block_size = config.threads;
    constexpr unsigned int     grid_size  = config.blocks;
    constexpr unsigned int     stride     = block_size * grid_size;
    // All threads of a generator must be in the same block
    static_assert(block_size % mt19937_octo_engine::threads_per_generator == 0,
                  "All eight threads of the generator must be in the same block");

    constexpr unsigned int input_width  = Distribution::input_width;
    constexpr unsigned int output_width = Distribution::output_width;

    const unsigned int thread_id = blockIdx.x * block_size + threadIdx.x;

    unsigned int input[input_width];
    T            output[output_width];

    // Generate one extra VecT if data is not aligned by sizeof(VecT) or
    // size % output_width != 0
    const unsigned int extra           = (head_size > 0 || tail_size > 0) ? 1 : 0;
    bool               is_extra_thread = false;

    // Engines have enough values, generated by the previous generate_long_kernel call,
    // but not yet used.
    // Since values are loaded from global memory (so dynamic indexing is not a problem),
    // it is beneficial to calculate what iterations will actually write data.
    const unsigned int j_start = start_input / stride;
    const unsigned int j_end   = (start_input + vec_size + extra + stride - 1) / stride;
    for(unsigned int j = j_start; j < j_end; j++)
    {
        if(j * stride + thread_id >= start_input
           && j * stride + thread_id - start_input < vec_size + extra)
        {
            mt19937_octo_engine_accessor<stride> accessor(engines);
#pragma unroll
            for(unsigned int i = 0; i < input_width; i++)
            {
                input[i] = mt19937_octo_engine::temper(
                    accessor.load_value(thread_id, j * input_width + i));
            }

            distribution(input, output);

            const size_t thread_index = j * stride + thread_id - start_input;

            // Mark an extra thread that will write head and tail
            is_extra_thread = thread_index == vec_size + extra - 1;

            if(thread_index < vec_size)
            {
                vec_data[thread_index] = *reinterpret_cast<VecT*>(output);
            }
        }
    }

    if constexpr(output_width > 1)
    {
        // Save head and tail, output was generated earlier
        if(is_extra_thread)
        {
            for(unsigned int o = 0; o < output_width; o++)
            {
                if(o < head_size)
                {
                    data[o] = output[o];
                }
                if(o > output_width - tail_size - 1)
                {
                    data[size + (output_width - tail_size - 1) - o] = output[o];
                }
            }
        }
    }
}

template<class ConfigProvider, bool IsDynamic, class T, class VecT, class Distribution>
ROCRAND_KERNEL __launch_bounds__((get_block_size<ConfigProvider, T>(
    IsDynamic))) void generate_long_kernel(unsigned int* __restrict__ engines,
                                           const unsigned int start_input,
                                           T* __restrict__ data,
                                           const size_t size,
                                           VecT* __restrict__ vec_data,
                                           const size_t       vec_size,
                                           const unsigned int head_size,
                                           const unsigned int tail_size,
                                           Distribution       distribution)
{
    constexpr generator_config config     = ConfigProvider::template device_config<T>(IsDynamic);
    constexpr unsigned int     block_size = config.threads;
    constexpr unsigned int     grid_size  = config.blocks;
    constexpr unsigned int     threads_per_generator = mt19937_octo_engine::threads_per_generator;
    // All threads of a generator must be in the same block
    static_assert(block_size % threads_per_generator == 0,
                  "All eight threads of the generator must be in the same block");

    constexpr unsigned int input_width      = Distribution::input_width;
    constexpr unsigned int output_width     = Distribution::output_width;
    constexpr unsigned int inputs_per_state
        = (mt19937_constants::n / mt19937_octo_engine::threads_per_generator) / input_width;
    constexpr unsigned int stride           = block_size * grid_size;
    constexpr unsigned int full_stride      = stride * inputs_per_state;

    const unsigned int thread_id = blockIdx.x * block_size + threadIdx.x;

    unsigned int input[input_width];
    T            output[output_width];

    // Workaround: since load() and store() use the same indices, the compiler decides to keep
    // computed addresses alive wasting 78 * 2 VGPRs. blockDim.x equals to block_size but it is
    // a runtime value so save() will compute new addresses.
    mt19937_octo_engine_accessor<stride> accessor(engines);
    mt19937_octo_engine engine = accessor.load(blockIdx.x * blockDim.x + threadIdx.x);

    size_t base_index = 0;

    // Start sequence: at least some engines have values, generated by the previous call for
    // the end sequence, but not yet used.
    if(start_input > 0)
    {
#pragma unroll
        for(unsigned int j = 0; j < inputs_per_state; j++)
        {
            // Skip used values
            if(j * stride + thread_id >= start_input)
            {
#pragma unroll
                for(unsigned int i = 0; i < input_width; i++)
                {
                    input[i] = mt19937_octo_engine::temper(engine.get(j * input_width + i));
                }

                distribution(input, output);

                const size_t thread_index = j * stride + thread_id - start_input;
                vec_data[thread_index]    = *reinterpret_cast<VecT*>(output);
            }
        }
        base_index = full_stride - start_input;
    }

    // Middle sequence: all engines write n * stride values together and use them all
    // in a fast unrolled loop.
    for(; base_index + full_stride <= vec_size; base_index += full_stride)
    {
        engine.gen_next_n();
#pragma unroll
        for(unsigned int j = 0; j < inputs_per_state; j++)
        {
#pragma unroll
            for(unsigned int i = 0; i < input_width; i++)
            {
                input[i] = mt19937_octo_engine::temper(engine.get(j * input_width + i));
            }

            distribution(input, output);

            const size_t thread_index = base_index + j * stride + thread_id;
            vec_data[thread_index]    = *reinterpret_cast<VecT*>(output);
        }
    }

    // Generate one extra VecT if data is not aligned by sizeof(VecT) or
    // size % output_width != 0
    const unsigned int extra = (head_size > 0 || tail_size > 0) ? 1 : 0;

    // End sequence: generate n values but use only a required part of them
    if(base_index < vec_size + extra)
    {
        bool is_extra_thread = false;
        engine.gen_next_n();
#pragma unroll
        for(unsigned int j = 0; j < inputs_per_state; j++)
        {
#pragma unroll
            for(unsigned int i = 0; i < input_width; i++)
            {
                input[i] = mt19937_octo_engine::temper(engine.get(j * input_width + i));
            }

            distribution(input, output);

            const size_t thread_index = base_index + j * stride + thread_id;

            // Mark an extra thread that will write head and tail
            is_extra_thread = thread_index == vec_size + extra - 1;

            if(thread_index < vec_size)
            {
                vec_data[thread_index] = *reinterpret_cast<VecT*>(output);
            }
            else
            {
                break;
            }
        }

        if constexpr(output_width > 1)
        {
            // Save head and tail, output was generated earlier
            if(is_extra_thread)
            {
                for(unsigned int o = 0; o < output_width; o++)
                {
                    if(o < head_size)
                    {
                        data[o] = output[o];
                    }
                    if(o > output_width - tail_size - 1)
                    {
                        data[size + (output_width - tail_size - 1) - o] = output[o];
                    }
                }
            }
        }
    }

    // save state
    accessor.save(thread_id, engine);
}

} // end namespace rocrand_host::detail

template<class ConfigProvider>
class rocrand_mt19937_template : public rocrand_generator_impl_base
{
public:
    using base_type        = rocrand_generator_impl_base;
    using octo_engine_type = ::rocrand_host::detail::mt19937_octo_engine;

    static constexpr inline unsigned int threads_per_generator
        = octo_engine_type::threads_per_generator;

    /// Number of threads per block for jump_ahead_kernel. Can be tweaked for performance.
    static constexpr inline unsigned int jump_ahead_thread_count = 128;

    rocrand_mt19937_template(unsigned long long seed   = 0,
                             rocrand_ordering   order  = ROCRAND_ORDERING_PSEUDO_DEFAULT,
                             hipStream_t        stream = 0)
        : base_type(order, 0, stream), m_seed(seed)
    {
        // Allocate device random number engines
        auto error = hipMalloc(&m_engines,
                               m_generator_count * rocrand_host::detail::mt19937_constants::n
                                   * sizeof(unsigned int));
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
    }

    rocrand_mt19937_template(const rocrand_mt19937_template&) = delete;

    rocrand_mt19937_template(rocrand_mt19937_template&& other)
        : base_type(other)
        , m_engines_initialized(std::exchange(other.m_engines_initialized, false))
        , m_engines(std::exchange(other.m_engines, nullptr))
        , m_start_input(other.m_start_input)
        , m_prev_input_width(other.m_prev_input_width)
        , m_seed(other.m_seed)
        , m_poisson(std::move(other.m_poisson))
        , m_generator_count(other.m_generator_count)
    {}

    rocrand_mt19937_template& operator=(const rocrand_mt19937_template&) = delete;

    rocrand_mt19937_template& operator=(rocrand_mt19937_template&& other)
    {
        *static_cast<base_type*>(this) = other;

        m_engines_initialized = std::exchange(other.m_engines_initialized, false);
        m_engines             = std::exchange(other.m_engines, nullptr);
        m_start_input         = other.m_start_input;
        m_prev_input_width    = other.m_prev_input_width;
        m_seed                = other.m_seed;
        m_poisson             = std::move(other.m_poisson);
        m_generator_count     = other.m_generator_count;

        return *this;
    }

    ~rocrand_mt19937_template()
    {
        if(m_engines != nullptr)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(m_engines));
        }
    }

    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_MT19937;
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
        (void)offset;
        // Can't set offset for MT19937
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
        if(m_engines_initialized)
        {
            return ROCRAND_STATUS_SUCCESS;
        }

        // TODO: make a version for generators that don't support per-type specialization of the configs and use that one
        // For now: just use the void config, assuming that all configs are the same
        rocrand_host::detail::generator_config config;
        hipError_t err = ConfigProvider::template host_config<void>(m_stream, m_order, config);
        if(err != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        m_generator_count = config.threads * config.blocks / threads_per_generator;

        if(m_engines != nullptr)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(m_engines));
        }
        // Allocate device random number engines
        err = hipMalloc(reinterpret_cast<void**>(&m_engines),
                        m_generator_count * rocrand_host::detail::mt19937_constants::n
                            * sizeof(unsigned int));
        if(err != hipSuccess)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        unsigned int* d_engines{};
        err = hipMalloc(&d_engines,
                        m_generator_count * rocrand_host::detail::mt19937_constants::n
                            * sizeof(unsigned int));
        if(err != hipSuccess)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        unsigned int* d_mt19937_jump{};
        err = hipMalloc(&d_mt19937_jump, sizeof(rocrand_h_mt19937_jump));
        if(err != hipSuccess)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_engines));
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        err = hipMemcpy(d_mt19937_jump,
                        rocrand_h_mt19937_jump,
                        sizeof(rocrand_h_mt19937_jump),
                        hipMemcpyHostToDevice);
        if(err != hipSuccess)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_engines));
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_mt19937_jump));
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        if(::rocrand_host::detail::is_ordering_dynamic(m_order))
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(rocrand_host::detail::jump_ahead_kernel<jump_ahead_thread_count, ConfigProvider, true>),
                dim3(m_generator_count),
                dim3(jump_ahead_thread_count),
                0,
                m_stream,
                d_engines,
                m_seed,
                d_mt19937_jump);
        }
        else
        {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(rocrand_host::detail::jump_ahead_kernel<jump_ahead_thread_count, ConfigProvider, false>),
                dim3(m_generator_count),
                dim3(jump_ahead_thread_count),
                0,
                m_stream,
                d_engines,
                m_seed,
                d_mt19937_jump);
        }

        err = hipStreamSynchronize(m_stream);
        if(err != hipSuccess)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_engines));
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_mt19937_jump));
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }

        err = hipFree(d_mt19937_jump);
        if(err != hipSuccess)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_engines));
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        // Assume all configs are the same and just use the one for unsigned int
        ROCRAND_LAUNCH_KERNEL_FOR_ORDERING(unsigned int,
                                           m_order,
                                           rocrand_host::detail::init_engines_kernel,
                                           dim3(config.blocks),
                                           dim3(config.threads),
                                           0,
                                           m_stream,
                                           m_engines,
                                           d_engines);

        err = hipStreamSynchronize(m_stream);
        if(err != hipSuccess)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(d_engines));
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }

        err = hipFree(d_engines);
        if(err != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        m_engines_initialized = true;
        m_start_input         = 0;
        m_prev_input_width    = 0;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T>>
    rocrand_status generate(T* data, size_t size, Distribution distribution = Distribution())
    {
        rocrand_status status = init();
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        constexpr unsigned int input_width  = Distribution::input_width;
        constexpr unsigned int output_width = Distribution::output_width;
        constexpr unsigned int inputs_per_state
            = (rocrand_host::detail::mt19937_constants::n / threads_per_generator) / input_width;
        const unsigned int stride      = threads_per_generator * m_generator_count;
        const unsigned int full_stride = stride * inputs_per_state;

        rocrand_host::detail::generator_config config;
        hipError_t err = ConfigProvider::template host_config<T>(m_stream, m_order, config);
        if(err != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }

        using vec_type = aligned_vec_type<T, output_width>;

        const uintptr_t uintptr = reinterpret_cast<uintptr_t>(data);
        const size_t    misalignment
            = (output_width - uintptr / sizeof(T) % output_width) % output_width;
        const unsigned int head_size = min(size, misalignment);
        const unsigned int tail_size = (size - head_size) % output_width;
        const size_t       vec_size  = (size - head_size) / output_width;

        // Generate one extra vec_type if data is not aligned by sizeof(vec_type) or
        // size % output_width != 0.
        // One extra output is enough for all types and distributions (output_width <= 2), except
        // uchar (output_width = 4): in very rare situations when both data and size are
        // misaligned, head and tail may be 2-3 and they may write 1-2 common values.
        const unsigned int extra = (head_size > 0 || tail_size > 0) ? 1 : 0;

        // Each iteration saves output_width values T as one vec_type.
        vec_type* vec_data = reinterpret_cast<vec_type*>(data + misalignment);

        if(m_prev_input_width != input_width && m_start_input > 0)
        {
            // Move to the next stride of inputs if input_width has changed so generators
            // will not use twice values used by the previous call. Some values may be discarded.

            // First we find the max number of values used by engines:
            const unsigned int max_used_engine_values
                = (m_start_input + stride - 1) / stride * m_prev_input_width;
            // and convert it to the number of inputs across all engines:
            m_start_input = (max_used_engine_values + input_width - 1) / input_width * stride;
            if(m_start_input >= full_stride)
            {
                m_start_input = 0;
            }
        }

        if(m_start_input > 0 && m_start_input + vec_size + extra <= full_stride)
        {
            // Engines have enough values, generated by the previous generate_long_kernel call.
            // This kernel does not load and store engines but loads values directly from global
            // memory.
            ROCRAND_LAUNCH_KERNEL_FOR_ORDERING(T,
                                               m_order,
                                               rocrand_host::detail::generate_short_kernel,
                                               dim3(config.blocks),
                                               dim3(config.threads),
                                               0,
                                               m_stream,
                                               m_engines,
                                               m_start_input,
                                               data,
                                               size,
                                               vec_data,
                                               vec_size,
                                               head_size,
                                               tail_size,
                                               distribution);
        }
        else
        {
            // There are not enough generated values or no values at all
            ROCRAND_LAUNCH_KERNEL_FOR_ORDERING(T,
                                               m_order,
                                               rocrand_host::detail::generate_long_kernel,
                                               dim3(config.blocks),
                                               dim3(config.threads),
                                               0,
                                               m_stream,
                                               m_engines,
                                               m_start_input,
                                               data,
                                               size,
                                               vec_data,
                                               vec_size,
                                               head_size,
                                               tail_size,
                                               distribution);
        }

        // check kernel status
        if(hipGetLastError() != hipSuccess)
        {
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }

        m_start_input      = (m_start_input + vec_size + extra) % full_stride;
        m_prev_input_width = input_width;

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
    rocrand_status generate_uniform(T* data, size_t data_size)
    {
        uniform_distribution<T> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T* data, size_t data_size, T mean, T stddev)
    {
        normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T* data, size_t data_size, T mean, T stddev)
    {
        log_normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    rocrand_status generate_poisson(unsigned int* data, size_t data_size, double lambda)
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
    bool          m_engines_initialized = false;
    unsigned int* m_engines             = nullptr;
    // The index of the next unused input across all engines (where "input" is `input_width`
    // unsigned int state values), it equals to the number of inputs used by previous generate
    // calls. 0 means that a new generation (gen_next_n) is required.
    unsigned int m_start_input;
    unsigned int m_prev_input_width;

    unsigned long long m_seed;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<> m_poisson;

    /// Number of independent generators. Value changes generated number stream.
    unsigned int m_generator_count = 0;
};

using rocrand_mt19937 = rocrand_mt19937_template<
    rocrand_host::detail::default_config_provider<ROCRAND_RNG_PSEUDO_MT19937>>;

#endif // ROCRAND_RNG_MT19937_H_
