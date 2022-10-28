// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_BENCHMARK_ROCRAND_UTILS_HPP_
#define ROCRAND_BENCHMARK_ROCRAND_UTILS_HPP_

#include "benchmark_utils.hpp"

#include <benchmark/benchmark.h>

#include <rocrand/rocrand.h>

#include <iostream>

#define ROCRAND_CHECK(condition)                                                               \
    do                                                                                         \
    {                                                                                          \
        rocrand_status status_ = condition;                                                    \
        if(status_ != ROCRAND_STATUS_SUCCESS)                                                  \
        {                                                                                      \
            std::cout << "ROCRAND error: " << status_ << " at " << __FILE__ << ":" << __LINE__ \
                      << std::endl;                                                            \
            exit(status_);                                                                     \
        }                                                                                      \
    }                                                                                          \
    while(0)

inline void add_common_benchmark_rocrand_info()
{
    int version;
    ROCRAND_CHECK(rocrand_get_version(&version));
    benchmark::AddCustomContext("rocrand_version", std::to_string(version));

    add_common_benchmark_info();
}

#endif // ROCRAND_BENCHMARK_ROCRAND_UTILS_HPP_
