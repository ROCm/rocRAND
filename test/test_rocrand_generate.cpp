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

#include "test_common.hpp"

class rocrand_generate_tests : public ::testing::TestWithParam<rocrand_rng_type> { };

TEST_P(rocrand_generate_tests, int_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator(
            &generator,
            rng_type
        )
    );

    const size_t size = 12563;
    unsigned int * data;
    HIP_CHECK(hipMalloc((void **)&data, size * sizeof(unsigned int)));
    HIP_CHECK(hipDeviceSynchronize());

    // Any sizes
    ROCRAND_CHECK(
        rocrand_generate(generator, data, 1)
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Any alignment
    ROCRAND_CHECK(
        rocrand_generate(generator, data+1, 2)
    );
    HIP_CHECK(hipDeviceSynchronize());

    ROCRAND_CHECK(
        rocrand_generate(generator, data, size)
    );
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST_P(rocrand_generate_tests, char_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator(
            &generator,
            rng_type
        )
    );

    const size_t size = 12563;
    unsigned char * data;
    HIP_CHECK(hipMalloc((void **)&data, size * sizeof(unsigned char)));
    HIP_CHECK(hipDeviceSynchronize());

    // Any sizes
    ROCRAND_CHECK(
        rocrand_generate_char(generator, data, 1)
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Any alignment
    ROCRAND_CHECK(
        rocrand_generate_char(generator, data+1, 2)
    );
    HIP_CHECK(hipDeviceSynchronize());

    ROCRAND_CHECK(
        rocrand_generate_char(generator, data, size)
    );
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST_P(rocrand_generate_tests, short_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator(
            &generator,
            rng_type
        )
    );

    const size_t size = 12563;
    unsigned short * data;
    HIP_CHECK(hipMalloc((void **)&data, size * sizeof(unsigned short)));
    HIP_CHECK(hipDeviceSynchronize());

    // Any sizes
    ROCRAND_CHECK(
        rocrand_generate_short(generator, data, 1)
    );
    HIP_CHECK(hipDeviceSynchronize());

    // Any alignment
    ROCRAND_CHECK(
        rocrand_generate_short(generator, data+1, 2)
    );
    HIP_CHECK(hipDeviceSynchronize());

    ROCRAND_CHECK(
        rocrand_generate_short(generator, data, size)
    );
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
}

TEST(rocrand_generate_tests, neg_test)
{
    const size_t size = 256;
    unsigned int * data = NULL;

    rocrand_generator generator = NULL;
    EXPECT_EQ(
        rocrand_generate(generator, (unsigned int *) data, size),
        ROCRAND_STATUS_NOT_CREATED
    );
}

INSTANTIATE_TEST_CASE_P(rocrand_generate_tests,
                        rocrand_generate_tests,
                        ::testing::ValuesIn(rng_types));
