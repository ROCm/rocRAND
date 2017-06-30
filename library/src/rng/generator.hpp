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

#ifndef ROCRAND_RNG_GENERATOR_H_
#define ROCRAND_RNG_GENERATOR_H_

#include <hip/hip_runtime.h>

#ifndef FQUALIFIERS
#define FQUALIFIERS __host__ __device__
#endif // FQUALIFIERS

#include <rocrand.h>
#include <rocrand_kernel.h>

#include "get_state_type.hpp"

// Just an example
// TODO: Implement
template <class state_type>
__global__ void generate_kernel(state_type * states, const size_t states_size,
                                unsigned int * data, const size_t n)
{
    // Kernel in which we use device functions that are equivalents
    // of curand_init(..), curand(..) etc. from curand_kernel.h
    state_type state = states[0];
    size_t id = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    if(id < n)
    {
        data[id] = rocrand(&state);
    }
}

// Base class for rocrand_generator_type
struct rocrand_generator_type_base { };

// rocRAND random number generator class template
template<rocrand_rng_type rng_type = ROCRAND_RNG_PSEUDO_PHILOX4_32_10>
struct rocrand_generator_type : public rocrand_generator_type_base
{
    typedef typename rocrand_get_state_type<rng_type>::type state_type;

    // ordering type
    size_t offset;
    hipStream_t stream;

    // PRNG states on device
    state_type * states;
    size_t states_size;

    rocrand_generator_type(hipStream_t stream = 0)
        : offset(0), stream(stream),
          states(0), states_size(0)
    {
        states_size = 1; // for example philox needs one state
        hipMalloc(&states, sizeof(state_type) * states_size);
    }

    ~rocrand_generator_type()
    {
        hipFree(static_cast<void*>(states));
    }

    /// Return generator's type
    constexpr rocrand_rng_type type() const
    {
        return rng_type;
    }

    template<class T>
    rocrand_status generate(T * data, size_t n)
    {
        // Just an example
        // TODO: Implement
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(generate_kernel),
            dim3(1), dim3(n), 0, stream,
            states, states_size, data, n
        );
        return ROCRAND_STATUS_SUCCESS;
    }
};

#endif // ROCRAND_GENERATOR_H_
