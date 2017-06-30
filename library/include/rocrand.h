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

#if !defined(ROCRAND_H_)
#define ROCRAND_H_

#include <hip/hip_runtime.h>

#ifndef ROCRANDAPI
#define ROCRANDAPI 
#endif

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */
    
/**
 * ROCRAND function call status types 
 */
enum rocrandStatus {
    ROCRAND_STATUS_SUCCESS = 0, ///< No errors
    ROCRAND_STATUS_VERSION_MISMATCH = 100, ///< Header file and linked library version do not match
    ROCRAND_STATUS_NOT_INITIALIZED = 101, ///< Generator not initialized
    ROCRAND_STATUS_ALLOCATION_FAILED = 102, ///< Memory allocation failed
    ROCRAND_STATUS_TYPE_ERROR = 103, ///< Generator is wrong type
    ROCRAND_STATUS_OUT_OF_RANGE = 104, ///< Argument out of range
    ROCRAND_STATUS_LENGTH_NOT_MULTIPLE = 105, ///< Length requested is not a multple of dimension
    ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106, ///< GPU does not have double precision required by MRG32k3a
    ROCRAND_STATUS_LAUNCH_FAILURE = 201, ///< Kernel launch failure
    ROCRAND_STATUS_PREEXISTING_FAILURE = 202, ///< Preexisting failure on library entry
    ROCRAND_STATUS_INTERNAL_ERROR = 999 ///< Internal library error
};
    
/**
 * ROCRAND generator types
 */
enum rocrandRngType {
    ROCRAND_RNG_PSEUDO_XORWOW = 100, ///< XORWOW pseudorandom generator
    ROCRAND_RNG_PSEUDO_MRG32K3A = 200, ///< MRG32k3a pseudorandom generator
    ROCRAND_RNG_PSEUDO_MTGP32 = 300, ///< Mersenne Twister MTGP32 pseudorandom generator
    ROCRAND_RNG_PSEUDO_PHILOX4_32_10 = 400, ///< PHILOX-4x32-10 pseudorandom generator
    ROCRAND_RNG_QUASI_SOBOL32 = 500, ///< Sobol32 quasirandom generator
};

    
#if defined(__cplusplus)
}
#endif /* __cplusplus */



#endif /* !defined(ROCRAND_H_) */