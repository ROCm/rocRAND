// Copyright (c) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.
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
#include "distributions.hpp"
#include "generator_type.hpp"
#include "system.hpp"

#include <rocrand/rocrand_scrambled_sobol32.h>
#include <rocrand/rocrand_scrambled_sobol64.h>
#include <rocrand/rocrand_sobol32.h>
#include <rocrand/rocrand_sobol64.h>

#include <rocrand/rocrand_scrambled_sobol32_constants.h>
#include <rocrand/rocrand_scrambled_sobol32_precomputed.h>
#include <rocrand/rocrand_scrambled_sobol64_constants.h>
#include <rocrand/rocrand_scrambled_sobol64_precomputed.h>
#include <rocrand/rocrand_sobol32_precomputed.h>
#include <rocrand/rocrand_sobol64_precomputed.h>

#include <rocrand/rocrand.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace rocrand_host::detail
{

template<unsigned int OutputPerThread,
         bool         Scrambled,
         class Engine,
         class Constant,
         class T,
         class Distribution>
__host__ __device__ void generate_sobol(dim3               block_idx,
                                        dim3               thread_idx,
                                        dim3               grid_dim,
                                        dim3               block_dim,
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

    const unsigned int dimension = block_idx.y;
    const unsigned int engine_id = block_idx.x * block_dim.x + thread_idx.x;
    const unsigned int stride    = grid_dim.x * block_dim.x;
    size_t             index     = engine_id;

    // Each thread of the current block uses the same direction vectors
    // (the dimension is determined by blockIdx.y)
    constexpr unsigned int vector_size = sizeof(Constant) == 4 ? 32 : 64;
    const Constant*        vectors_ptr = [=]
    {
        if constexpr(use_shared_vectors)
        {
#ifdef __HIP_PLATFORM_AMD__
            // On AMD GPUs we must use a constexpr size shared array for performance.
            // But this code won't compile with NVCC, because we are in a __host__ __device__
            // function.
            __shared__ Constant shared_vectors[vector_size];
#else
            // NVCC won't accept extern __shared__ Constant shared_bytes[];
            // Thereby we must resort to aliasing.
            extern __shared__ unsigned char shared_bytes[];
            auto* shared_vectors = reinterpret_cast<Constant*>(&shared_bytes[0]);
#endif
            if(thread_idx.x < vector_size)
            {
                shared_vectors[thread_idx.x]
                    = direction_vectors[dimension * vector_size + thread_idx.x];
            }
            syncthreads<true>{}();
            return shared_vectors;
        }
        else
        {
            return direction_vectors + dimension * vector_size;
        }
    }();

    const Constant scramble_constant = Scrambled ? scramble_constants[dimension] : 0;
    const auto     create_engine
        = [scramble_constant](const Constant* vectors, const unsigned int offset)
    {
        if constexpr(Scrambled)
        {
            return Engine(vectors, scramble_constant, offset);
        }
        else
        {
            (void)scramble_constant;
            return Engine(vectors, offset);
        }
    };

    data += dimension * n;

    // All distributions generate one output from one input
    // Generation of, for example, 2 shorts from 1 uint or
    // 2 floats from 2 uints using Box-Muller transformation
    // is impossible because the resulting sequence is not
    // quasi-random anymore.

    if constexpr(output_per_thread == 1)
    {
        const unsigned int engine_offset = engine_id * output_per_thread;
        Engine             engine        = create_engine(vectors_ptr, offset + engine_offset);

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
        Engine engine = create_engine(vectors_ptr, offset + engine_offset);

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

/// \brief Selects the appropriate device engine type for the template parameters.
template<bool Is64, bool Scrambled, bool UseSharedVectors>
using sobol_device_engine_t = typename sobol_device_engine<Is64, Scrambled, UseSharedVectors>::type;

/// \brief Loads the appropriate direction vectors and scramble constants (if applicable) for the
/// Sobol generator specified by the template arguments.
/// \tparam System The system type, e.g. device or host system. See system.hpp for details.
/// \tparam Is64 Whether the generator output is 64 bits.
/// \tparam Scrambled Whether the generator is scrambled Sobol.
template<class System, bool Is64, bool Scrambled>
class sobol_constant_accessor
{
public:
    using constant_type = std::conditional_t<Is64, unsigned long long int, unsigned int>;
    using system_type   = System;

    sobol_constant_accessor()
    {
        m_status = allocate_direction_vectors(&m_direction_vectors);
        if(m_status != ROCRAND_STATUS_SUCCESS)
        {
            return;
        }
        m_status = allocate_scramble_constants(&m_scramble_constants);
    }

    sobol_constant_accessor(sobol_constant_accessor&& other)
        : m_status(std::exchange(other.m_status, ROCRAND_STATUS_SUCCESS))
        , m_direction_vectors(std::exchange(other.m_direction_vectors, nullptr))
        , m_scramble_constants(std::exchange(other.m_scramble_constants, nullptr))
    {}

    sobol_constant_accessor(const sobol_constant_accessor&) = delete;

    sobol_constant_accessor& operator=(sobol_constant_accessor&& other)
    {
        m_status             = std::exchange(other.m_status, ROCRAND_STATUS_SUCCESS);
        m_direction_vectors  = std::exchange(other.m_direction_vectors, nullptr);
        m_scramble_constants = std::exchange(other.m_scramble_constants, nullptr);
        return *this;
    }

    sobol_constant_accessor& operator=(const sobol_constant_accessor&) = delete;

    ~sobol_constant_accessor()
    {
        deallocate();
    }

    rocrand_status get_direction_vectors(const constant_type** direction_vectors) const
    {
        *direction_vectors = m_direction_vectors;
        return m_status;
    }

    rocrand_status get_scramble_constants(const constant_type** scramble_constants) const
    {
        *scramble_constants = m_scramble_constants;
        return m_status;
    }

private:
    rocrand_status m_status             = ROCRAND_STATUS_SUCCESS;
    constant_type* m_direction_vectors  = nullptr;
    constant_type* m_scramble_constants = nullptr;

    static const constant_type* get_direction_vectors_ptr()
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

    // Device
    template<bool IsDevice = system_type::is_device()>
    static std::enable_if_t<IsDevice, rocrand_status>
        allocate_direction_vectors(constant_type** direction_vectors)
    {
        constexpr size_t direction_vectors_count = Is64 ? SOBOL64_N : SOBOL32_N;
        constexpr size_t direction_vectors_bytes = sizeof(constant_type) * direction_vectors_count;
        const rocrand_status status
            = system_type::alloc(direction_vectors, direction_vectors_count);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }
        const constant_type* h_direction_vectors = get_direction_vectors_ptr();
        const hipError_t     error               = hipMemcpy(*direction_vectors,
                                           h_direction_vectors,
                                           direction_vectors_bytes,
                                           hipMemcpyHostToDevice);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    // Not device
    template<bool IsDevice = system_type::is_device()>
    static std::enable_if_t<!IsDevice, rocrand_status>
        allocate_direction_vectors(constant_type** direction_vectors)
    {
        *direction_vectors = const_cast<constant_type*>(get_direction_vectors_ptr());
        return ROCRAND_STATUS_SUCCESS;
    }

    static const constant_type* get_scramble_constants_ptr()
    {
        if constexpr(Is64)
        {
            return h_scrambled_sobol64_constants;
        }
        else
        {
            return h_scrambled_sobol32_constants;
        }
    }

    // Not scrambled
    template<bool IsScrambled = Scrambled>
    static std::enable_if_t<!IsScrambled, rocrand_status>
        allocate_scramble_constants(constant_type** scramble_constants)
    {
        *scramble_constants = nullptr;
        return ROCRAND_STATUS_SUCCESS;
    }

    // Scrambled, on device
    template<bool IsScrambled = Scrambled, bool IsDevice = system_type::is_device()>
    static std::enable_if_t<IsScrambled && IsDevice, rocrand_status>
        allocate_scramble_constants(constant_type** scramble_constants)
    {
        const rocrand_status status = system_type::alloc(scramble_constants, SCRAMBLED_SOBOL_DIM);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
        }
        const hipError_t error = hipMemcpy(*scramble_constants,
                                           get_scramble_constants_ptr(),
                                           sizeof(constant_type) * SCRAMBLED_SOBOL_DIM,
                                           hipMemcpyHostToDevice);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_INTERNAL_ERROR;
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    // Scrambled, not on device
    template<bool IsScrambled = Scrambled, bool IsDevice = system_type::is_device()>
    static std::enable_if_t<IsScrambled && !IsDevice, rocrand_status>
        allocate_scramble_constants(constant_type** scramble_constants)
    {
        *scramble_constants = const_cast<constant_type*>(get_scramble_constants_ptr());
        return ROCRAND_STATUS_SUCCESS;
    }

    // Not device
    template<bool IsDevice = system_type::is_device()>
    std::enable_if_t<!IsDevice> deallocate()
    {}

    // Device, not scrambled
    template<bool IsDevice = system_type::is_device(), bool IsScrambled = Scrambled>
    std::enable_if_t<IsDevice && !IsScrambled> deallocate()
    {
        system_type::free(m_direction_vectors);
    }

    // Device, scrambled
    template<bool IsDevice = system_type::is_device(), bool IsScrambled = Scrambled>
    std::enable_if_t<IsDevice && IsScrambled> deallocate()
    {
        system_type::free(m_direction_vectors);
        system_type::free(m_scramble_constants);
    }
};

} // end namespace rocrand_host::detail

template<class System, bool Is64, bool Scrambled>
class rocrand_sobol_template : public rocrand_generator_impl_base
{
public:
    static constexpr inline bool is_scrambled = Scrambled;
    using system_type                         = System;
    using base_type                           = rocrand_generator_impl_base;
    using engine_type
        = ::rocrand_host::detail::sobol_device_engine_t<Is64, Scrambled, system_type::is_device()>;
    using constant_type = std::conditional_t<Is64, unsigned long long int, unsigned int>;
    using constant_accessor
        = rocrand_host::detail::sobol_constant_accessor<system_type, Is64, Scrambled>;

    rocrand_sobol_template(unsigned long long offset = 0,
                           rocrand_ordering   order  = ROCRAND_ORDERING_QUASI_DEFAULT,
                           hipStream_t        stream = 0)
        : base_type(order, offset, stream)
    {
        rocrand_status status = get_constants().get_direction_vectors(&m_direction_vectors);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            throw status;
        }
        status = get_constants().get_direction_vectors(&m_scramble_constants);
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            throw status;
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
        constexpr uint32_t shared_mem_bytes
#ifdef __HIP_PLATFORM_AMD__
            // On AMD GPUs we must resort to static shared memory for performance.
            = 0;
#else
            = system_type::is_device() ? ((Is64 ? 64 : 32) * sizeof(constant_type)) : 0;
#endif

        const size_t       size             = data_size / m_dimensions;
        constexpr uint32_t output_per_block = threads * output_per_thread;
        const uint32_t     blocks
            = std::min(max_blocks,
                       static_cast<uint32_t>((size + output_per_block - 1) / output_per_block));

        // blocks_x must be power of 2 because strided discard (leap frog)
        // supports only power of 2 jumps
        const uint32_t blocks_x = next_power2((blocks + m_dimensions - 1) / m_dimensions);
        const uint32_t blocks_y = m_dimensions;

        using block_size_provider
            = rocrand_host::detail::static_block_size_config_provider<threads>;

        status
            = system_type::template launch<rocrand_host::detail::generate_sobol<output_per_thread,
                                                                                Scrambled,
                                                                                engine_type,
                                                                                constant_type,
                                                                                T,
                                                                                Distribution>,
                                           block_size_provider>(dim3(blocks_x, blocks_y),
                                                                dim3(threads),
                                                                shared_mem_bytes,
                                                                m_stream,
                                                                data,
                                                                size,
                                                                m_direction_vectors,
                                                                m_scramble_constants,
                                                                m_current_offset,
                                                                distribution);
        // Check kernel status
        if(status != ROCRAND_STATUS_SUCCESS)
        {
            return status;
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
    static const constant_accessor& get_constants()
    {
        // Every instance of each Sobol variant shares the constants.
        // The initialization of accessor happens only at the first invocation of this function.
        static constant_accessor accessor;
        return accessor;
    }

    bool                 m_initialized        = false;
    unsigned int         m_dimensions         = 1;
    unsigned int         m_current_offset     = 0;
    const constant_type* m_direction_vectors  = nullptr;
    const constant_type* m_scramble_constants = nullptr;

    // For caching of Poisson for consecutive generations with the same lambda
    poisson_distribution_manager<ROCRAND_DISCRETE_METHOD_CDF, !system_type::is_device()> m_poisson;

    // m_offset from base_type

    static size_t next_power2(size_t x)
    {
        size_t power = 1;
        while(power < x)
        {
            power *= 2;
        }
        return power;
    }
};

using rocrand_sobol32           = rocrand_sobol_template<rocrand_system_device, false, false>;
using rocrand_sobol64           = rocrand_sobol_template<rocrand_system_device, true, false>;
using rocrand_scrambled_sobol32 = rocrand_sobol_template<rocrand_system_device, false, true>;
using rocrand_scrambled_sobol64 = rocrand_sobol_template<rocrand_system_device, true, true>;
template<bool UseHostFunc>
using rocrand_sobol32_host = rocrand_sobol_template<rocrand_system_host<UseHostFunc>, false, false>;
template<bool UseHostFunc>
using rocrand_sobol64_host = rocrand_sobol_template<rocrand_system_host<UseHostFunc>, true, false>;
template<bool UseHostFunc>
using rocrand_scrambled_sobol32_host
    = rocrand_sobol_template<rocrand_system_host<UseHostFunc>, false, true>;
template<bool UseHostFunc>
using rocrand_scrambled_sobol64_host
    = rocrand_sobol_template<rocrand_system_host<UseHostFunc>, true, true>;

#endif // ROCRAND_RNG_SOBOL_H_
