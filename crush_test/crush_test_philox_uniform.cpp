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
#include <file_common.hpp>

#include <hip/hip_runtime.h>
#include <rocrand.h>

extern "C" 
{
#include "bbattery.h"
}

int main(void)
{
    rocrand_generator generator;
    rocrand_create_generator(
        &generator,
        ROCRAND_RNG_PSEUDO_PHILOX4_32_10
    );

    const size_t size = 50000;
    unsigned int * d_data;
    unsigned int * h_data = new unsigned int[size];

    hipMalloc((void **)&d_data, size * sizeof(unsigned int));
    hipDeviceSynchronize();

    rocrand_generate(generator, (unsigned int *) d_data, size);
    hipDeviceSynchronize();
    
    hipMemcpy(h_data, d_data, size * sizeof(unsigned int), hipMemcpyDeviceToHost);
    hipDeviceSynchronize();

    rocrand_file_write_results("philox_uniform.txt", h_data, size);
    
    bbattery_SmallCrushFile("philox_uniform.txt");
    
    hipFree(d_data);
    delete[] h_data;
    rocrand_destroy_generator(generator);
    
    return 0;
}