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

#include <hip/hip_runtime.h>
#include <rocrand.h>

#include <rng/generator_type.hpp>

#define HIP_CHECK(state) ASSERT_EQ(state, hipSuccess)

TEST(rocrand_generator_type_tests, rocrand_generator)
{
    rocrand_generator g = NULL;
    EXPECT_EQ(g, static_cast<rocrand_generator>(0));

    g = new rocrand_generator_type<>;
    rocrand_generator_type<> * gg = static_cast<rocrand_generator_type<>* >(g);
    EXPECT_NE(gg, static_cast<rocrand_generator>(0));
    EXPECT_EQ(gg->type(), ROCRAND_RNG_PSEUDO_PHILOX4_32_10);
    EXPECT_EQ(gg->get_offset(), 0ULL);
    EXPECT_EQ(gg->get_seed(), 0ULL);
    EXPECT_EQ(gg->get_stream(), (hipStream_t)(0));
    delete(g);
}

TEST(rocrand_generator_type_tests, default_ctor_test)
{
    rocrand_generator_type<> g;
    EXPECT_EQ(g.type(), ROCRAND_RNG_PSEUDO_PHILOX4_32_10);
    EXPECT_EQ(g.get_offset(), 0ULL);
    EXPECT_EQ(g.get_seed(), 0ULL);
    EXPECT_EQ(g.get_stream(), (hipStream_t)(0));
}

TEST(rocrand_generator_type_tests, ctor_test)
{
    rocrand_generator_type<ROCRAND_RNG_PSEUDO_XORWOW> g;
    EXPECT_EQ(g.type(), ROCRAND_RNG_PSEUDO_XORWOW);
    EXPECT_EQ(g.get_seed(), 0ULL);
    EXPECT_EQ(g.get_seed(), 0ULL);
    EXPECT_EQ(g.get_stream(), (hipStream_t)(0));
}

TEST(rocrand_generator_type_tests, set_stream_test)
{
    rocrand_generator_type<ROCRAND_RNG_PSEUDO_PHILOX4_32_10> g;
    EXPECT_EQ(g.get_stream(), (hipStream_t)(0));
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    g.set_stream(stream);
    EXPECT_EQ(g.get_stream(), stream);
    g.set_stream(NULL);
    EXPECT_EQ(g.get_stream(), (hipStream_t)(0));
    HIP_CHECK(hipStreamDestroy(stream));
}
