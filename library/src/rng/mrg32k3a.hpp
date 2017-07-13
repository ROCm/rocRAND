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

#ifndef FQUALIFIERS
#define FQUALIFIERS __host__ __device__
#endif // FQUALIFIERS

#include <rocrand.h>

#include "mrg32k3a_state.hpp"
#include "generator_type.hpp"
#include "distributions.hpp"

namespace rocrand_mrg32k3a_detail
{
    template <
        class Type, class StateType,
        class Generator, class Distribution
    >
    __global__
    void generate_kernel(StateType * states,
                         bool init_states,
                         unsigned long long seed,
                         unsigned long long offset,
                         Type * data, const size_t n,
                         Generator generator,
                         Distribution distribution)
    {
        const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = state_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        // Load or init the state
        StateType state;
        if(init_states)
        {
            generator.init_state(&state, offset, index, seed);
        }
        else
        {
            state = states[state_id];
        }
        
        while(index < n)
        {
            data[index] = distribution(generator(&state));
            index += stride;
        }
        
        // Save state
        states[state_id] = state;
    }

    template <
        class RealType, class StateType,
        class Generator, class Distribution
    >
    __global__
    void generate_normal_kernel(StateType * states,
                                bool init_states,
                                unsigned long long seed,
                                unsigned long long offset,
                                RealType * data, const size_t n,
                                Generator generator,
                                Distribution distribution)
    {
        // TODO: Implement it 
    }

    // Returns 1 value
    template<class Generator, class StateType>
    struct generator_state_wrapper
    {
        __host__ __device__
        generator_state_wrapper(Generator generator, StateType state)
            : generator(generator), state(state) {}
        
        __host__ __device__
        ~generator_state_wrapper() {}

        __forceinline__ __host__ __device__
        unsigned long long operator()()
        {
            return generator(&state);
        }

        Generator generator;
        StateType state;
    };

    template <
        class StateType,
        class Generator, class Distribution
    >
    __global__
    void generate_poisson_kernel(StateType * states,
                                 bool init_states,
                                 unsigned long long seed,
                                 unsigned long long offset,
                                 unsigned int * data, const size_t n,
                                 Generator generator,
                                 Distribution distribution)
    {
        // TODO: Implement it 
    }

} // end namespace rocrand_mrg32k3a_detail

class rocrand_mrg32k3a : public rocrand_generator_type<ROCRAND_RNG_PSEUDO_MRG32K3A>
{
public:
    using base_type = rocrand_generator_type<ROCRAND_RNG_PSEUDO_MRG32K3A>;
    using state_type = base_type::state_type;

    class mrg32k3a_generator
    {
    public:
        __forceinline__ __host__ __device__
        unsigned long long operator()(state_type * state)
        {
            unsigned long long p;
            unsigned long long * g1 = state->g1;
            unsigned long long * g2 = state->g2;
            
            p = ROCRAND_RNG_MRG32K3A_A12 * g1[1] + ROCRAND_RNG_MRG32K3A_A13N 
                * (ROCRAND_RNG_MRG32K3A_M1 - g1[0]);
            p = (p & (ROCRAND_RNG_MRG32K3A_POW32 - 1)) + (p >> 32) 
                * ROCRAND_RNG_MRG32K3A_M1C;
            if (p >= ROCRAND_RNG_MRG32K3A_M1) 
                p -= ROCRAND_RNG_MRG32K3A_M1;

            g1[0] = g1[1]; g1[1] = g1[2]; g1[2] = p;

            p = ROCRAND_RNG_MRG32K3A_A21 * g2[2] + ROCRAND_RNG_MRG32K3A_A23N 
                * (ROCRAND_RNG_MRG32K3A_M2 - g2[0]);
            p = (p & (ROCRAND_RNG_MRG32K3A_POW32 - 1)) + (p >> 32) 
                * ROCRAND_RNG_MRG32K3A_M2C;
            p = (p & (ROCRAND_RNG_MRG32K3A_POW32 - 1)) + (p >> 32) 
                * ROCRAND_RNG_MRG32K3A_M2C;
            if (p >= ROCRAND_RNG_MRG32K3A_M2) 
                p -= ROCRAND_RNG_MRG32K3A_M2;
                
            g2[0] = g2[1]; g2[1] = g2[2]; g2[2] = p;

            p = g1[2] - g2[2];
            if (g1[2] <= g2[2]) 
                p += ROCRAND_RNG_MRG32K3A_M1;  // 0 < p <= M1
            
            return p;
        }

        __forceinline__ __host__ __device__
        void init_state(state_type * state,
                        unsigned long long offset,
                        unsigned long long sequence,
                        unsigned long long seed)
        {
            state->set_seed(seed);
            state->discard_sequence(sequence);
            state->discard(offset);

        }

        __forceinline__ __host__ __device__
        void discard(state_type * state, unsigned int n)
        {
            state->discard(n);
        }

        __forceinline__ __host__ __device__
        void discard(state_type * state)
        {
            state->discard();
        }

        __forceinline__ __host__ __device__
        void discard_sequence(state_type * state, unsigned int n)
        {
            state->discard_sequence(n);
        }
    };

    rocrand_mrg32k3a(unsigned long long seed = 12345,
                     unsigned long long offset = 1,
                     hipStream_t stream = 0)
        : base_type(seed, offset, stream),
          m_states_initialized(false), m_states(NULL), m_states_size(1024 * 256)
    {
        auto error = hipMalloc(&m_states, sizeof(state_type) * m_states_size);
        if(error != hipSuccess)
        {
            throw ROCRAND_STATUS_ALLOCATION_FAILED;
        }
    }

    ~rocrand_mrg32k3a()
    {
        hipFree(m_states);
    }

    void reset()
    {
        m_states_initialized = false;
    }

    /// Changes seed to \p seed and resets generator state.
    void set_seed(unsigned long long seed)
    {
        m_seed = seed;
        m_states_initialized = false;
    }

    void set_offset(unsigned long long offset)
    {
        m_offset = offset;
        m_states_initialized = false;
    }

    template<class T, class Distribution = mrg_uniform_distribution<T> >
    rocrand_status generate(T * data, size_t data_size,
                            const Distribution& distribution = Distribution())
    {
        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 1024;
        #endif
        const uint32_t blocks = max_blocks;

        namespace detail = rocrand_mrg32k3a_detail;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::generate_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_states, !m_states_initialized, m_seed, m_offset,
            data, data_size, m_generator, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_states_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T * data, size_t n)
    {
        mrg_uniform_distribution<T> udistribution;
        return generate(data, n, udistribution);
    }

    template<class T>
    rocrand_status generate_normal(T * data, size_t data_size, T stddev, T mean)
    {
        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 1024;
        #endif
        const uint32_t blocks = max_blocks;

        normal_distribution<T> distribution(mean, stddev);

        namespace detail = rocrand_mrg32k3a_detail;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::generate_normal_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_states, !m_states_initialized, m_seed, m_offset,
            data, data_size, m_generator, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_states_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_log_normal(T * data, size_t data_size, T stddev, T mean)
    {
        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 1024;
        #endif
        const uint32_t blocks = max_blocks;

        log_normal_distribution<T> distribution(mean, stddev);

        namespace detail = rocrand_mrg32k3a_detail;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::generate_normal_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_states, !m_states_initialized, m_seed, m_offset,
            data, data_size, m_generator, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_states_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

    rocrand_status generate_poisson(unsigned int * data, size_t data_size, double lambda)
    {
        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128; // 512
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 1024;
        #endif
        const uint32_t blocks = max_blocks;

        poisson_distribution<unsigned int> distribution(lambda);

        namespace detail = rocrand_mrg32k3a_detail;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::generate_poisson_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_states, !m_states_initialized, m_seed, m_offset,
            data, data_size, m_generator, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_states_initialized = true;
        return ROCRAND_STATUS_SUCCESS;
    }

private:
    bool m_states_initialized;
    state_type * m_states;
    size_t m_states_size;

    // m_seed from base_type
    // m_offset from base_type
    mrg32k3a_generator m_generator;
};

#endif // ROCRAND_RNG_MRG32K3A_H_
