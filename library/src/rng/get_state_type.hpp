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

#ifndef ROCRAND_RNG_GET_STATE_TYPE_H_
#define ROCRAND_RNG_GET_STATE_TYPE_H_

#include <hip/hip_runtime.h>

#include <rocrand.h>
#include "states.hpp"

template<rocrand_rng_type>
struct rocrand_get_state_type
{
    typedef void type;
};

template<>
struct rocrand_get_state_type<ROCRAND_RNG_PSEUDO_PHILOX4_32_10>
{
    typedef rocrand_philox4_32_10_state type;
};

template<>
struct rocrand_get_state_type<ROCRAND_RNG_PSEUDO_XORWOW>
{
    typedef rocrand_xorwow_state type;
};

#endif // ROCRAND_RNG_GET_STATE_TYPE_H_


