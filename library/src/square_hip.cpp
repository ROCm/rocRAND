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

#include "square_hip.h"

/**
    @ingroup device
    @brief Squares element in array and writes to output array
    
    Template kernel that squares each element of input array and writes to output array
			
	@param C_d Output array
	@param A_d Input array
	@param N Length of array
*/
template <typename T>
__global__ void
vector_square(hipLaunchParm lp, T *C_d, const T *A_d, size_t N)
{
    size_t offset = (hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x);
    size_t stride = hipBlockDim_x * hipGridDim_x ;

    for (size_t i=offset; i<N; i+=stride) {
        C_d[i] = A_d[i] * A_d[i];
    }
}

/**
    @ingroup host
    @brief Host function to call square kernel
    
    Sample function that demonstrates hip functions to allocate memory, read/write and to call the device kernel
*/
void test()
{
    float *A_d, *C_d;
    float *A_h, *C_h;
    size_t N = 1000000;
    size_t Nbytes = N * sizeof(float);

    hipDeviceProp_t props;
    CHECK(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);

    printf ("info: allocate host mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    A_h = (float*)malloc(Nbytes);
    CHECK(A_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    C_h = (float*)malloc(Nbytes);
    CHECK(C_h == 0 ? hipErrorMemoryAllocation : hipSuccess );
    // Fill with Phi + i
    for (size_t i=0; i<N; i++)
    {
        A_h[i] = 1.618f + i;
    }

    printf ("info: allocate device mem (%6.2f MB)\n", 2*Nbytes/1024.0/1024.0);
    CHECK(hipMalloc(&A_d, Nbytes));
    CHECK(hipMalloc(&C_d, Nbytes));


    printf ("info: copy Host2Device\n");
    CHECK ( hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));

    const unsigned blocks = 512;
    const unsigned threadsPerBlock = 256;

    printf ("info: launch 'vector_square' kernel\n");
    hipLaunchKernel(vector_square, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, N);

    printf ("info: copy Device2Host\n");
    CHECK ( hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));

    printf ("info: check result\n");
    for (size_t i=0; i<N; i++)  {
        if (C_h[i] != A_h[i] * A_h[i]) {
            CHECK(hipErrorUnknown);
        }
    }
    printf ("PASSED!\n");
}
