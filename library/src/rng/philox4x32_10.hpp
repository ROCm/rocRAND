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

#ifndef ROCRAND_RNG_PHILOX4X32_10_H_
#define ROCRAND_RNG_PHILOX4X32_10_H_

#include <hip/hip_runtime.h>

#ifndef FQUALIFIERS
#define FQUALIFIERS __host__ __device__
#endif // FQUALIFIERS

#include <rocrand.h>

#include "philox4x32_10_state.hpp"
#include "generator_type.hpp"

namespace detail
{
    template <class state_type>
    __global__
    void generate_uniform_kernel(state_type * states, const size_t states_size,
                                 unsigned int * data, const size_t n);

    typedef
        ::rocrand_generator_type<ROCRAND_RNG_PSEUDO_PHILOX4_32_10>::state_type
        rocrand_philox4x32_10_state_type;


    __global__
    void generate_uniform_kernel(rocrand_philox4x32_10_state_type * states,
                                 const size_t states_size,
                                 unsigned int * data,
                                 const size_t n)
    {
        // rocrand_philox4x32_10_state_type state = states[0];
        size_t id = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
        if(id < n)
        {
            data[id] = 43210; // !!!!! HERE WE HAVE TO CALL DEVICE FUNCTIONS THAT
            // TAKES POINTER TO rocrand_philox4x32_10::state_type AND RETURNS uint
        }
    }
}

struct rocrand_philox4x32_10 : public ::rocrand_generator_type<ROCRAND_RNG_PSEUDO_PHILOX4_32_10>
{
    typedef typename ::rocrand_generator_type<ROCRAND_RNG_PSEUDO_PHILOX4_32_10> base_type;
    typedef typename base_type::state_type state_type;

    rocrand_philox4x32_10(unsigned long long offset = 0, hipStream_t stream = 0)
        : base_type(offset, stream), m_states(NULL), m_states_size(0)
    {

    }

    ~rocrand_philox4x32_10()
    {

    }

    template<class T>
    rocrand_status generate_uniform(T * data, size_t n)
    {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(detail::generate_uniform_kernel),
            dim3(1), dim3(n), 0, stream,
            m_states, m_states_size,
            data, n
        );
        return ROCRAND_STATUS_SUCCESS;
    }

    template<class T>
    rocrand_status generate(T * data, size_t n)
    {
        return generate_uniform(data, n);
    }

private:
    state_type * m_states;
    unsigned long m_states_size;
};

#endif // ROCRAND_RNG_PHILOX4X32_10_H_
