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

#include <hip/hip_runtime.h>

// nextafterf(float, float) is not implemented for device on HIP/HCC platform
#if defined(__HIP_PLATFORM_HCC__) && defined(__HIP_DEVICE_COMPILE__)
__forceinline__ __device__
float nextafterf(const float from, const float to)
{
    if(from == to) return to;
    // (2.3283064e-10f) is float min value
    return fminf(from + (2.3283064e-10f), to);
}
#endif // defined(__HIP_PLATFORM_HCC__) && defined(__HIP_DEVICE_COMPILE__)

#endif // ROCRAND_RNG_COMMON_H_
