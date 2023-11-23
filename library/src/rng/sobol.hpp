// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_RNG_SOBOL_H_
#define ROCRAND_RNG_SOBOL_H_

#include "common.hpp"
#include "config_types.hpp"
#include "device_engines.hpp"
#include "distributions.hpp"
#include "generator_type.hpp"

#include <rocrand/rocrand.h>
#include <rocrand/rocrand_scrambled_sobol32_constants.h>
#include <rocrand/rocrand_scrambled_sobol32_precomputed.h>
#include <rocrand/rocrand_scrambled_sobol64_constants.h>
#include <rocrand/rocrand_scrambled_sobol64_precomputed.h>
#include <rocrand/rocrand_sobol32_precomputed.h>
#include <rocrand/rocrand_sobol64_precomputed.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

namespace rocrand_host::detail
{

template<unsigned int OutputPerThread,
         bool         Scrambled,
         class Engine,
         class Constant,
         class T,
         class Distribution>
ROCRAND_KERNEL __launch_bounds__(ROCRAND_DEFAULT_MAX_BLOCK_SIZE) void generate_kernel(
    T*                 data,
    const size_t       n,
    const Constant*    direction_vectors,
    const Constant*    scramble_constants,
    const unsigned int offset,
    Distribution       distribution)
{
    constexpr unsigned int output_per_thread  = OutputPerThread;
    constexpr bool         use_shared_vectors = Engine::uses_shared_vectors();
    using vec_type                            = aligned_vec_type<T, output_per_thread>;

    const unsigned int dimension = blockIdx.y;
    const unsigned int engine_id = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int stride    = gridDim.x * blockDim.x;
    size_t             index     = engine_id;

    // Each thread of the current block uses the same direction vectors
    // (the dimension is determined by blockIdx.y)
    constexpr unsigned int vector_size = sizeof(Constant) == 4 ? 32 : 64;
    const Constant*        vectors_ptr;
    if constexpr(use_shared_vectors)
    {
        __shared__ Constant shared_vectors[vector_size];
        if(threadIdx.x < vector_size)
        {
            shared_vectors[threadIdx.x] = direction_vectors[dimension * vector_size + threadIdx.x];
        }
        __syncthreads();
        vectors_ptr = shared_vectors;
    }
    else
    {
        vectors_ptr = direction_vectors + dimension * vector_size;
    }

    const Constant scramble_constant = Scrambled ? scramble_constants[dimension] : 0;
    const auto     create_engine     = [scramble_constant, vectors_ptr](const unsigned int offset)
    {
        if constexpr(Scrambled)
        {
            return Engine(vectors_ptr, scramble_constant, offset);
        }
        else
        {
            (void)scramble_constant;
            return Engine(vectors_ptr, offset);
        }
    };

    data += dimension * n;

    // All distributions generate one output from one input
    // Generation of, for example, 2 shorts from 1 uint or
    // 2 floats from 2 uints using Box-Muller transformation
    // is impossible because the resulting sequence is not
    // quasi-random anymore.

    if(output_per_thread == 1)
    {
        const unsigned int engine_offset = engine_id * output_per_thread;
        Engine             engine        = create_engine(offset + engine_offset);

        while(index < n)
        {
            data[index] = distribution(engine.current());
            engine.discard_stride(stride);
            index += stride;
        }
    }
    else
    {
        const uintptr_t uintptr   = reinterpret_cast<uintptr_t>(data);
        const size_t misalignment = (output_per_thread - uintptr / sizeof(T)) % output_per_thread;
        const unsigned int head_size = min(n, misalignment);
        const unsigned int tail_size = (n - head_size) % output_per_thread;
        const size_t       vec_n     = (n - head_size) / output_per_thread;

        const unsigned int engine_offset
            = engine_id * output_per_thread
              + (engine_id == 0 ? 0 : head_size); // The first engine writes head_size values
        Engine engine = create_engine(offset + engine_offset);

        if(engine_id == 0)
        {
            // If data is not aligned by sizeof(vec_type)
            for(unsigned int o = 0; o < head_size; o++)
            {
                data[o] = distribution(engine.current());
                engine.discard();
            }
        }

        vec_type* vec_data = reinterpret_cast<vec_type*>(data + misalignment);
        while(index < vec_n)
        {
            Engine engine_copy = engine;

            T output[output_per_thread];
            for(unsigned int i = 0; i < output_per_thread; i++)
            {
                output[i] = distribution(engine.current());
                engine.discard();
            }

            vec_data[index] = *reinterpret_cast<vec_type*>(output);

            // Restore from a copy and use fast discard_stride with power of 2 stride
            engine = engine_copy;
            engine.discard_stride(stride * output_per_thread);
            index += stride;
        }

        if(index == vec_n)
        {
            // Fill tail values (up to output_per_thread-1)
            for(unsigned int o = 0; o < tail_size; o++)
            {
                data[n - tail_size + o] = distribution(engine.current());
                engine.discard();
            }
        }
    }
}

template<bool Is64, bool Scrambled, bool UseSharedVectors>
struct sobol_device_engine;

template<bool UseSharedVectors>
struct sobol_device_engine<false, false, UseSharedVectors>
{
    using type = ::rocrand_device::sobol32_engine<UseSharedVectors>;
};

template<bool UseSharedVectors>
struct sobol_device_engine<false, true, UseSharedVectors>
{
    using type = ::rocrand_device::scrambled_sobol32_engine<UseSharedVectors>;
};

template<bool UseSharedVectors>
struct sobol_device_engine<true, false, UseSharedVectors>
{
    using type = ::rocrand_device::sobol64_engine<UseSharedVectors>;
};

template<bool UseSharedVectors>
struct sobol_device_engine<true, true, UseSharedVectors>
{
    using type = ::rocrand_device::scrambled_sobol64_engine<UseSharedVectors>;
};

template<bool Is64, bool Scrambled, bool UseSharedVectors>
using sobol_device_engine_t = typename sobol_device_engine<Is64, Scrambled, UseSharedVectors>::type;

} // end namespace rocrand_host::detail

template<bool Is64, bool Scrambled>
class rocrand_sobol : public rocrand_generator_impl_base
{
public:
    static constexpr inline bool is_scrambled = Scrambled;
    using base_type                           = rocrand_generator_impl_base;
    using engine_type   = ::rocrand_host::detail::sobol_device_engine_t<Is64, Scrambled, true>;
    using constant_type = std::conditional_t<Is64, unsigned long long int, unsigned int>;

    rocrand_sobol(unsigned long long offset = 0,
                  rocrand_ordering   order  = ROCRAND_ORDERING_QUASI_DEFAULT,
                  hipStream_t        stream = 0)
        : base_type(order, offset, stream)
        , m_initialized(false)
        , m_dimensions(1)
        , m_scramble_constants(nullptr)
    {
        // Allocate direction vectors
        constexpr size_t direction_vectors_bytes
            = sizeof(constant_type) * (Is64 ? SOBOL64_N : SOBOL32_N);
        hipError_t error;
        error = hipMalloc(&m_direction_vectors, direction_vectors_bytes);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }

        const constant_type* direction_vectors = get_direction_vectors();
        error                                  = hipMemcpy(m_direction_vectors,
                          direction_vectors,
                          direction_vectors_bytes,
                          hipMemcpyHostToDevice);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_INTERNAL_ERROR;
        }

        if constexpr(Scrambled)
        {
            // Allocate scramble constants
            error = hipMalloc(&m_scramble_constants, sizeof(constant_type) * SCRAMBLED_SOBOL_DIM);
            if(error != hipSuccess)
            {
                throw ROCRAND_STATUS_ALLOCATION_FAILED;
            }
            const void* scrambled_constants_source
                = Is64 ? static_cast<const void*>(h_scrambled_sobol64_constants)
                       : static_cast<const void*>(h_scrambled_sobol32_constants);
            error = hipMemcpy(m_scramble_constants,
                              scrambled_constants_source,
                              sizeof(constant_type) * SCRAMBLED_SOBOL_DIM,
                              hipMemcpyHostToDevice);
            if(error != hipSuccess)
            {
                throw ROCRAND_STATUS_INTERNAL_ERROR;
            }
        }
    }

    rocrand_sobol(const rocrand_sobol&) = delete;

    rocrand_sobol(rocrand_sobol&&) = delete;

    rocrand_sobol& operator=(const rocrand_sobol&) = delete;

    rocrand_sobol& operator=(rocrand_sobol&&) = delete;

    ~rocrand_sobol()
    {
        ROCRAND_HIP_FATAL_ASSERT(hipFree(m_direction_vectors));
        if constexpr(Scrambled)
        {
            ROCRAND_HIP_FATAL_ASSERT(hipFree(m_scramble_constants));
        }
    }

    static constexpr rocrand_rng_type type()
    {
        if constexpr(Is64)
        {
            if constexpr(Scrambled)
            {
                return ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
            }
            else
            {
                return ROCRAND_RNG_QUASI_SOBOL64;
            }
        }
        else
        {
            if constexpr(Scrambled)
            {
                return ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
            }
            else
            {
                return ROCRAND_RNG_QUASI_SOBOL32;
            }
        }
    }

    void reset() override final
    {
        m_initialized = false;
    }

    void set_seed(unsigned long long seed)
    {
        (void)seed;
    }

    unsigned long long get_seed() const
    {
        return 0;
    }

    rocrand_status set_order(rocrand_ordering order)
    {
        if(!rocrand_host::detail::is_ordering_quasi(order))
        {
            return ROCRAND_STATUS_OUT_OF_RANGE;
        }
        m_order = order;
        reset();
        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status set_dimensions(unsigned int dimensions) override final
    {
        constexpr unsigned int max_dimensions = Scrambled ? SCRAMBLED_SOBOL_DIM : SOBOL_DIM;
        if(dimensions < 1 || dimensions > max_dimensions)
        {
            return ROCRAND_STATUS_OUT_OF_RANGE;
        }

        m_dimensions  = dimensions;
        m_initialized = false;

        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status init()
    {
        if(m_initialized)
        {
            return ROCRAND_STATUS_SUCCESS;
        }

        m_current_offset = static_cast<unsigned int>(m_offset);
        m_initialized    = true;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = sobol_uniform_distribution<T>>
    rocrand_status generate(T* data, size_t data_size, Distribution distribution = Distribution())
    {
        if constexpr(!Is64 && std::is_same_v<T, unsigned long long int>)
        {
            return ROCRAND_STATUS_TYPE_ERROR;
        }

        constexpr unsigned int output_per_thread
            = sizeof(T) >= sizeof(int) ? 1 : sizeof(int) / sizeof(T);

        if(data_size % m_dimensions != 0)
        {
            return ROCRAND_STATUS_LENGTH_NOT_MULTIPLE;
        }

        rocrand_status status = init();
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }

        constexpr uint32_t threads    = 256;
        constexpr uint32_t max_blocks = 4096;

        const size_t       size             = data_size / m_dimensions;
        constexpr uint32_t output_per_block = threads * output_per_thread;
        const uint32_t     blocks
            = std::min(max_blocks,
                       static_cast<uint32_t>((size + output_per_block - 1) / output_per_block));

        // blocks_x must be power of 2 because strided discard (leap frog)
        // supports only power of 2 jumps
        const uint32_t blocks_x = next_power2((blocks + m_dimensions - 1) / m_dimensions);
        const uint32_t blocks_y = m_dimensions;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(
                rocrand_host::detail::generate_kernel<output_per_thread, Scrambled, engine_type>),
            dim3(blocks_x, blocks_y),
            dim3(threads),
            0,
            m_stream,
            data,
            size,
            m_direction_vectors,
            m_scramble_constants,
            m_current_offset,
            distribution);
        // Check kernel status
        if(hipGetLastError() != hipSuccess)
        {
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }

        m_current_offset += size;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T* data, size_t data_size)
    {
        sobol_uniform_distribution<T> distribution;
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_normal(T* data, size_t data_size, T mean, T stddev)
    {
        sobol_normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_log_normal(T* data, size_t data_size, T mean, T stddev)
    {
        sobol_log_normal_distribution<T> distribution(mean, stddev);
        return generate(data, data_size, distribution);
    }

    template<class T>
    rocrand_status generate_poisson(T* data, size_t data_size, double lambda)
    {
        static_assert(Is64 || std::is_same_v<T, uint32_t>,
                      "The 32 bit sobol generator can only generate 32bit poisson");
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
    static const auto* get_direction_vectors()
    {
        if constexpr(Is64)
        {
            if constexpr(Scrambled)
            {
                return rocrand_h_scrambled_sobol64_direction_vectors;
            }
            else
            {
                return rocrand_h_sobol64_direction_vectors;
            }
        }
        else
        {
            if constexpr(Scrambled)
            {
                return rocrand_h_scrambled_sobol32_direction_vectors;
            }
            else
            {
                return rocrand_h_sobol32_direction_vectors;
            }
        }
    }

    bool           m_initialized;
    unsigned int   m_dimensions;
    unsigned int   m_current_offset;
    constant_type* m_direction_vectors;
    constant_type* m_scramble_constants;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<ROCRAND_DISCRETE_METHOD_CDF> m_poisson;

    // m_offset from base_type

    size_t next_power2(size_t x)
    {
        size_t power = 1;
        while(power < x)
        {
            power *= 2;
        }
        return power;
    }
};

using rocrand_sobol32           = rocrand_sobol<false, false>;
using rocrand_sobol64           = rocrand_sobol<true, false>;
using rocrand_scrambled_sobol32 = rocrand_sobol<false, true>;
using rocrand_scrambled_sobol64 = rocrand_sobol<true, true>;

#endif // ROCRAND_RNG_SOBOL_H_
