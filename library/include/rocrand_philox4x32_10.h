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

#ifndef ROCRAND_PHILOX4X32_10_H_
#define ROCRAND_PHILOX4X32_10_H_

#include <rocrand.h>

#ifndef FQUALIFIERS
#define FQUALIFIERS __device__
#endif // FQUALIFIERS

struct rocrand_state_philox4x32_10
{
    uint4 counter;
    uint4 output;
    uint2 key;
    unsigned int state;
};

FQUALIFIERS unsigned int rocrand(rocrand_state_philox4x32_10 * state)
{
    (void) *state;
    return 43210;
}

FQUALIFIERS uint4 rocrand4(rocrand_state_philox4x32_10 * state)
{
    (void) *state;
    return uint4{ 43210, 43210, 43210, 43210 };
}

FQUALIFIERS void skipahead(unsigned long long offset, rocrand_state_philox4x32_10 * state)
{
    (void) offset;
    (void) *state;
}

FQUALIFIERS void skipahead_subsequence(unsigned long long offset, rocrand_state_philox4x32_10 * state)
{
    (void) offset;
    (void) *state;
}

#endif // ROCRAND_PHILOX4X32_10_H_
