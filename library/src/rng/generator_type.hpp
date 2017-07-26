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

#ifndef ROCRAND_RNG_GENERATOR_TYPE_H_
#define ROCRAND_RNG_GENERATOR_TYPE_H_

#include <hip/hip_runtime.h>
#include <rocrand.h>

struct rocrand_generator_base_type
{
    rocrand_generator_base_type(rocrand_rng_type rng_type) : rng_type(rng_type) {}
    const rocrand_rng_type rng_type;

    virtual ~rocrand_generator_base_type() {}
};

// rocRAND random number generator base class
template<rocrand_rng_type GeneratorType = ROCRAND_RNG_PSEUDO_PHILOX4_32_10>
struct rocrand_generator_type : public rocrand_generator_base_type
{
    using base_type = rocrand_generator_base_type;

    rocrand_generator_type(unsigned long long seed = 0,
                           unsigned long long offset = 0,
                           hipStream_t stream = 0)
        : base_type(GeneratorType),
          m_seed(seed), m_offset(offset), m_stream(stream)
    {

    }

    /// Return generator's type
    constexpr rocrand_rng_type type() const
    {
        return rng_type;
    }

    unsigned long long get_seed() const
    {
        return m_seed;
    }

    unsigned long long get_offset() const
    {
        return m_offset;
    }

    hipStream_t get_stream() const
    {
        return m_stream;
    }

    void set_stream(hipStream_t stream)
    {
        m_stream = stream;
    }

protected:
    // ordering type
    unsigned long long m_seed;
    unsigned long long m_offset;
    hipStream_t m_stream;
};

#endif // ROCRAND_RNG_GENERATOR_TYPE_H_
