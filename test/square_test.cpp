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

#include <stdio.h>
#include <gtest/gtest.h>

#include "square_hip.h"

#define HIP_1D_KERNEL_LOOP(i, n)                               \
  for (int i = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x; \
       i < (n);                                                \
       i += hipBlockDim_x * hipGridDim_x)


#define NUM_THREADS 256

template <typename T, int blockSize>
__global__ void reduce(T *g_idata, T *g_odata, int n) {
    __shared__ T sdata[blockSize];
    int i = hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;

    sdata[hipThreadIdx_x] = g_idata[i];

    __syncthreads();
    // do reduction in shared mem
    for (int s=1; s < hipBlockDim_x; s *=2) {
        int index = 2 * s * hipThreadIdx_x;;

        if (index < hipBlockDim_x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (hipThreadIdx_x == 0) g_odata[hipBlockIdx_x] = sdata[0];
}

TEST(SquareTest, hipRANDCall)
{
    EXPECT_NO_THROW(
        test(); // call hipRAND function
    );
}

TEST(SquareTest, SimpleTest)
{
    int x = 1;
    EXPECT_EQ(1, x);
    x++;
    EXPECT_EQ(2, x);
}
