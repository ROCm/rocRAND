// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "parity.hpp"

#include <iostream>
#include <vector>

#ifndef DEFAULT_RAND_N
    #define DEFAULT_RAND_N (1024 * 1024 * 128)
#endif

// FIND OUT:
// - Shared generators to test -- done
// - Shared distributions to test (defer)
// - Skipping (?)
// - Seeds (easy)
// - generating a stream from a state and then generating another stream (continuity) (?)

const char* generator_name(const generator_type rng_type)
{
    switch(rng_type)
    {
    case generator_type::XORWOW:
        return "xorwow";
    case generator_type::MRG32K3A:
        return "mrg32k3a";
    case generator_type::MTGP32:
        return "mtgp32";
    case generator_type::PHILOX4_32_10:
        return "philox";
    case generator_type::MT19937:
        return "mt19937";
    case generator_type::SOBOL32:
        return "sobol32";
    case generator_type::SCRAMBLED_SOBOL32:
        return "scrambled_sobol32";
    case generator_type::SOBOL64:
        return "sobol64";
    case generator_type::SCRAMBLED_SOBOL64:
        return "scrambled_sobol64";
    }
}

bool generator_is_64bit(const generator_type rng_type)
{
    switch(rng_type)
    {
    case generator_type::SOBOL64:
    case generator_type::SCRAMBLED_SOBOL64:
        return true;
    default:
        return false;
    }
}

template<typename T, typename F, typename G>
void run_test(const test_case& test_case, F generate_rocrand, G generate_curand)
{
    constexpr size_t n = DEFAULT_RAND_N;

    std::cout << generator_name(test_case.rng_type) << "(size=" << test_case.size;

    if (generator_is_psuedo(test_case.rng_type))
    {
        if(test_case.prng_seed < 0)
            std::cout << ", seed=default";
        else
            std::cout << ", seed=" << test_case.prng_seed;
    }
    else
    {
        if(test_case.qrng_dimensions < 0)
            std::cout << ", dims=default";
        else
            std::cout << ", dims=" << test_case.qrng_dimensions;
    }

    if(test_case.offset < 0)
        std::cout << ", offset=default";
    else
        std::cout << ", offset=" << test_case.offset;
    std::cout << "): ";

    const std::vector<T> rocrand_results = generate_rocrand(test_case);
    const std::vector<T> curand_results = generate_curand(test_case);

    size_t i = 0;
    for(; i < n; ++i)
    {
        if (rocrand_results[i] != curand_results[i])
        {
            break;
        }
    }

    if(i < n)
    {
        std::cout << "mismatch at index " << i << std::endl;
    }
    else
    {
        std::cout << "results are equal" << std::endl;
    }
}

void run_case(const test_case& test_case)
{
    if (generator_is_64bit(test_case.rng_type))
    {
        run_test<unsigned long long>(test_case,
                                     test_rocrand_generate_long_long,
                                     test_curand_generate_long_long);
    }
    else
    {
        run_test<unsigned int>(test_case,
                               test_rocrand_generate,
                               test_curand_generate);
    }

    // TODO: Other distributions?
}

void test_offset(const generator_type rng_type, size_t size, long long offset)
{
    if (generator_is_psuedo(rng_type))
    {
        for(const long long seed : {-1ll, 12345ll})
        {
            test_case test_case;
            test_case.rng_type = rng_type;
            test_case.size = size;
            test_case.prng_seed = seed;
            test_case.offset = offset;
            run_case(test_case);
        }
    }
    else
    {
        for(const long long dims : {1ll, 3ll})
        {
            test_case test_case;
            test_case.rng_type = rng_type;
            // Round down to multiple of dimensions
            test_case.size = size - size % dims;
            test_case.qrng_dimensions = dims;
            test_case.offset = offset;

            test_case.qrng_dimensions = dims;
            run_case(test_case);
        }
    }
}

int main() {
    constexpr generator_type generator_types[] =
    {
        generator_type::XORWOW,
        generator_type::MRG32K3A,
        generator_type::MTGP32,
        generator_type::PHILOX4_32_10,
        // generator_type::MT19937, // not supported when USE_HIP_CPU
        generator_type::SOBOL32,
        generator_type::SCRAMBLED_SOBOL32,
        generator_type::SOBOL64,
        generator_type::SCRAMBLED_SOBOL64
    };

    for(const generator_type rng_type : generator_types)
    {
        for(const long long offset : {-1ll, 112121116104111110ll})
        {
            test_offset(rng_type, DEFAULT_RAND_N, offset);
            if (rng_type == generator_type::MTGP32 || rng_type == generator_type::MT19937)
                break;
        }
    }
}
