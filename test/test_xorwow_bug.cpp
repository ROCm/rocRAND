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

#include <vector>
#include <cmath>

#include <hip/hip_runtime.h>

#include <rocrand_xorwow_precomputed.h>

#define HIP_CHECK(x) ASSERT_EQ(x, hipSuccess)
#define HIPRAND_CHECK(state) ASSERT_EQ(state, HIPRAND_STATUS_SUCCESS)

#ifdef __AMDGCN__
#define READ_EXEC() __builtin_amdgcn_read_exec()
#else
#define READ_EXEC() -1
#endif

typedef uint64_t exec_mask_t;

__global__
void kernel1(unsigned int * output, const size_t size, unsigned long long * execs)
{
    const unsigned int state_id = hipThreadIdx_x;
    const unsigned int global_size = hipBlockDim_x;

    unsigned int subsequence = state_id;
    // unsigned int subsequence = state_id ^ 1;
    unsigned int x = 0;
    unsigned int mi = 0;
    while (subsequence > 0)
    {
        const unsigned int * m = d_xorwow_sequence_jump_matrices[mi];
        for (int ii = 0; ii < (subsequence & 3); ii++)
        {
            // #pragma unroll 1
            for (int i = 0; i < XORWOW_N; i++)
            {
                // #pragma unroll 1
                for (int j = 0; j < XORWOW_M; j++)
                {
                    for (int k = 0; k < XORWOW_N; k++)
                    {
                        x ^= m[i * XORWOW_M * XORWOW_N + j * XORWOW_N + k];
                    }
                }
            }
        }
        mi++;
        subsequence >>= 2;
    }

    unsigned int index = state_id;
    unsigned int i = 0;
    while(index < size)
    {
        {
            exec_mask_t r = READ_EXEC();
            if (state_id == 2) execs[i] = r;
        }

        output[index] = x + i;
        index += global_size;
        i++;
    }
}

TEST(xorwow_bug, test1)
{
    const size_t output_size = 250;
    const size_t block = 64;

    unsigned int * output;
    HIP_CHECK(hipMalloc((void **)&output, output_size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<unsigned int> output_host(output_size, 12345);
    HIP_CHECK(
        hipMemcpy(
            output, output_host.data(),
            output_size * sizeof(unsigned int),
            hipMemcpyHostToDevice
        )
    );
    HIP_CHECK(hipDeviceSynchronize());

    unsigned long long * execs;
    HIP_CHECK(hipMalloc((void **)&execs, 64 * sizeof(unsigned long long)));

    std::vector<unsigned long long> execs_host(64, 0);
    HIP_CHECK(
        hipMemcpy(
            execs, execs_host.data(),
            64 * sizeof(unsigned long long),
            hipMemcpyHostToDevice
        )
    );

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(kernel1),
        dim3(1), dim3(block), 0, 0,
        output, output_size, execs
    );
    HIP_CHECK(hipPeekAtLastError());

    HIP_CHECK(
        hipMemcpy(
            output_host.data(), output,
            output_size * sizeof(unsigned int),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(output));

    for(size_t i = 0; i < output_size; i++)
    {
        const unsigned int v = output_host[i];
        std::cout << i << " " << v << (v == 12345 ? "!!!!!!!!!!!!!!!" : "") << '\n';
    }
    std::cout << std::endl;

    HIP_CHECK(
        hipMemcpy(
            execs_host.data(), execs,
            64 * sizeof(unsigned long long),
            hipMemcpyDeviceToHost
        )
    );
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipFree(execs));

    std::cout << "exec:" << '\n';
    for(size_t i = 0; i < (output_size + block - 1) / block; i++)
    {
        std::cout << std::hex << std::setw(16) << execs_host[i] << '\n';
    }

    for(auto v : output_host)
    {
        EXPECT_NE(v, 12345);
    }
}
