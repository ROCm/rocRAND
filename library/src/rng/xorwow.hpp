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

#ifndef ROCRAND_RNG_XORWOW_H_
#define ROCRAND_RNG_XORWOW_H_

namespace rocrand_xorwow_detail
{
    template <class StateType, class Generator>
    __global__
    void init_states_kernel(StateType * states,
                            unsigned long long seed,
                            unsigned long long offset,
                            Generator generator)
    {
        const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

        StateType state;
        generator.init_state(&state, offset, state_id, seed);

        states[state_id] = state;
    }

    template <
        class Type, class StateType,
        class Generator, class Distribution
    >
    __global__
    void generate_kernel(StateType * states,
                         Type * data, const size_t n,
                         Generator generator,
                         Distribution distribution)
    {
        const unsigned int state_id = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
        unsigned int index = state_id;
        unsigned int stride = hipGridDim_x * hipBlockDim_x;

        StateType state = states[state_id];

        while(index < n)
        {
            data[index] = distribution(generator(&state));
            index += stride;
        }

        states[state_id] = state;
    }
}

class rocrand_xorwow : public rocrand_generator_type<ROCRAND_RNG_PSEUDO_XORWOW>
{
public:
    using base_type = rocrand_generator_type<ROCRAND_RNG_PSEUDO_XORWOW>;
    using state_type = base_type::state_type;

    class xorwow_generator
    {
    public:

        __forceinline__ __host__ __device__
        unsigned int operator()(state_type * state)
        {
            state->discard();
            return (*state)();
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
    };

    rocrand_xorwow(unsigned long long seed = 0,
                   unsigned long long offset = 0,
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

    ~rocrand_xorwow()
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

    rocrand_status init()
    {
        if (m_states_initialized)
            return ROCRAND_STATUS_SUCCESS;

        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128;
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 1024;
        #endif
        const uint32_t blocks = max_blocks;

        namespace detail = rocrand_xorwow_detail;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::init_states_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_states, m_seed, m_offset, m_generator
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        m_states_initialized = true;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T, class Distribution = uniform_distribution<T> >
    rocrand_status generate(T * data, size_t data_size,
                            const Distribution& distribution = Distribution())
    {
        rocrand_status status = init();
        if (status != ROCRAND_STATUS_SUCCESS)
            return status;

        #ifdef __HIP_PLATFORM_NVCC__
        const uint32_t threads = 128;
        const uint32_t max_blocks = 128;
        #else
        const uint32_t threads = 256;
        const uint32_t max_blocks = 1024;
        #endif
        const uint32_t blocks = max_blocks;

        namespace detail = rocrand_xorwow_detail;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::generate_kernel),
            dim3(blocks), dim3(threads), 0, m_stream,
            m_states,
            data, data_size, m_generator, distribution
        );
        // Check kernel status
        if(hipPeekAtLastError() != hipSuccess)
            return ROCRAND_STATUS_LAUNCH_FAILURE;

        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate_uniform(T * data, size_t n)
    {
        uniform_distribution<T> udistribution;
        return generate(data, n, udistribution);
    }

private:
    bool m_states_initialized;
    state_type * m_states;
    size_t m_states_size;

    // m_seed from base_type
    // m_offset from base_type
    xorwow_generator m_generator;
};

#endif // ROCRAND_RNG_XORWOW_H_
