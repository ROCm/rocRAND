// Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.
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
#include <string>

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

inline std::string engine_name(const rocrand_rng_type rng_type)
{
    switch(rng_type)
    {
        case ROCRAND_RNG_PSEUDO_MTGP32: return "mtgp32";
        case ROCRAND_RNG_PSEUDO_MT19937: return "mt19937";
        case ROCRAND_RNG_PSEUDO_XORWOW: return "xorwow";
        case ROCRAND_RNG_PSEUDO_MRG31K3P: return "mrg31k3p";
        case ROCRAND_RNG_PSEUDO_MRG32K3A: return "mrg32k3a";
        case ROCRAND_RNG_PSEUDO_PHILOX4_32_10: return "philox";
        case ROCRAND_RNG_PSEUDO_LFSR113: return "lfsr113";
        case ROCRAND_RNG_PSEUDO_THREEFRY2_32_20: return "threefry2x32";
        case ROCRAND_RNG_PSEUDO_THREEFRY2_64_20: return "threefry2x64";
        case ROCRAND_RNG_PSEUDO_THREEFRY4_32_20: return "threefry4x32";
        case ROCRAND_RNG_PSEUDO_THREEFRY4_64_20: return "threefry4x64";
        case ROCRAND_RNG_QUASI_SOBOL32: return "sobol32";
        case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32: return "scrambled_sobol32";
        case ROCRAND_RNG_QUASI_SOBOL64: return "sobol64";
        case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64: return "scrambled_sobol64";
        default: return "unknown_rng_type";
    }
}

struct benchmark_config
{
    std::size_t bytes{};
    double      lambda{};
};

#endif // ROCRAND_BENCHMARK_ROCRAND_UTILS_HPP_
