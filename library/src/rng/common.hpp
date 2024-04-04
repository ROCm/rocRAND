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

#ifndef ROCRAND_RNG_COMMON_H_
#define ROCRAND_RNG_COMMON_H_

#include <hip/hip_runtime.h>

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__ __host__
#endif

#if !defined(USE_DEVICE_DISPATCH) && !defined(_WIN32) && defined(__HIP_PLATFORM_AMD__)
    #define USE_DEVICE_DISPATCH
#endif

#include <rocrand/rocrand_common.h>

#include <hip/hip_runtime_api.h>

#include <cstdio>
#include <cstdlib>

template<class T, unsigned int N>
struct alignas(sizeof(T) * N) aligned_vec_type
{
    T data[N];
};

/**
 * \brief Check for a HIP error and exit the program if encountered.
 *
 * This should only be used where other error reporting mechanism cannot be used, e.g. in
 * destructors, where throwing is not an option.
 */
#define ROCRAND_HIP_FATAL_ASSERT(expression)                                     \
    do                                                                           \
    {                                                                            \
        hipError_t _error = (expression);                                        \
        if(_error != hipSuccess)                                                 \
        {                                                                        \
            std::fprintf(stderr,                                                 \
                         "rocRAND internal error: %s in function %s at %s:%d\n", \
                         hipGetErrorName(_error),                                \
                         __func__,                                               \
                         __FILE__,                                               \
                         __LINE__);                                              \
            std::abort();                                                        \
        }                                                                        \
    }                                                                            \
    while(false)

#endif // ROCRAND_RNG_COMMON_H_
