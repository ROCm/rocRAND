// Copyright (c) 2019 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef TEST_COMMON_HPP_
#define TEST_COMMON_HPP_

#include <cstdlib>

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)
#define ROCRAND_CHECK(state) ASSERT_EQ(state, ROCRAND_STATUS_SUCCESS)

#define HIP_CHECK_NON_VOID(condition)         \
{                                    \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
}
const rocrand_rng_type rng_types[] = {
    ROCRAND_RNG_PSEUDO_PHILOX4_32_10,
    ROCRAND_RNG_PSEUDO_MRG32K3A,
    ROCRAND_RNG_PSEUDO_XORWOW,
    ROCRAND_RNG_PSEUDO_MTGP32,
    ROCRAND_RNG_QUASI_SOBOL32
};

bool supports_hmm()
{
    hipDeviceProp_t device_prop;
    int device_id;
    HIP_CHECK_NON_VOID(hipGetDevice(&device_id));
    HIP_CHECK_NON_VOID(hipGetDeviceProperties(&device_prop, device_id));
    if (device_prop.managedMemory == 1) return true;

    return false;
}

bool use_hmm()
{
    if (getenv("ROCRAND_USE_HMM") == nullptr)
    {
        return false;
    }

    if (strcmp(getenv("ROCRAND_USE_HMM"), "1") == 0)
    {
        return true;
    }
    return false;
}

// Helper for HMM allocations: if device supports managedMemory, and HMM is requested through
// ROCRAND_MALLOC_MANAGED environment variable
template <class T>
hipError_t hipMallocHelper(T** devPtr, size_t size)
{
    if (use_hmm())
    {
        return hipMallocManaged((void**)devPtr, size);
    }
    else
    {
        return hipMalloc((void**)devPtr, size);
    }
    return hipSuccess;
}


#endif // TEST_COMMON_HPP_
