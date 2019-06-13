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

#ifndef ROCRAND_RNG_COMMON_H_
#define ROCRAND_RNG_COMMON_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__ __host__
#endif

#include <hip/hip_fp16.h>

#include <rocrand/rocrand_common.h>

struct __attribute__((__packed__)) rocrand_half2
{
    __half x;
    __half y;

    FQUALIFIERS
    rocrand_half2() = default;

    FQUALIFIERS
    ~rocrand_half2() = default;


    FQUALIFIERS
    rocrand_half2(const __half x,
                  const __half y) : x(x), y(y)
    {
    }

    #if __hcc_major__ < 1 || __hcc_major__ == 1 && __hcc_minor__ < 2
    FQUALIFIERS
    rocrand_half2& operator =(const rocrand_half2& h2)
    {
        x = h2.x;
        y = h2.y;
        return *this;
    }
    #endif
};

struct __attribute__((__packed__)) rocrand_half4
{
    __half x;
    __half y;
    __half z;
    __half w;

    FQUALIFIERS
    rocrand_half4() = default;

    FQUALIFIERS
    ~rocrand_half4() = default;


    FQUALIFIERS
    rocrand_half4(const __half x,
                  const __half y,
                  const __half z,
                  const __half w) : x(x), y(y), z(z), w(w)
    {
    }

    #if __hcc_major__ < 1 || __hcc_major__ == 1 && __hcc_minor__ < 2
    FQUALIFIERS
    rocrand_half4& operator =(const rocrand_half4& h4)
    {
        x = h4.x;
        y = h4.y;
        z = h4.z;
        w = h4.w;
        return *this;
    }
    #endif
};

#endif // ROCRAND_RNG_COMMON_H_
