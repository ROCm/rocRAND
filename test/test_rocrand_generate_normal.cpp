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

TEST(rocrand_generate_normal_tests, float_test)
{
    rocrand_generator generator;
    ASSERT_EQ(
        rocrand_create_generator(
            &generator,
            ROCRAND_RNG_PSEUDO_PHILOX4_32_10
        ),
        ROCRAND_STATUS_SUCCESS
    );

    const size_t size = 256;
    float mean = 5.0f;
    float stddev = 2.0f;
    float * data;
    ASSERT_EQ(
        hipMalloc((void **)&data, size * sizeof(float)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // n must be even
    EXPECT_EQ(
        rocrand_generate_normal(generator, (float *) data, 1, mean, stddev),
        ROCRAND_STATUS_LENGTH_NOT_MULTIPLE
    );

    // pointer must be aligned
    EXPECT_EQ(
        rocrand_generate_normal(generator, (float *)(data+1), 2, mean, stddev),
        ROCRAND_STATUS_LENGTH_NOT_MULTIPLE
    );

    EXPECT_EQ(
        rocrand_generate_normal(generator, (float *) data, size, mean, stddev),
        ROCRAND_STATUS_SUCCESS
    );

    EXPECT_EQ(hipFree(data), hipSuccess);
    EXPECT_EQ(
        rocrand_destroy_generator(generator),
        ROCRAND_STATUS_SUCCESS
    );
}

TEST(rocrand_generate_normal_tests, double_test)
{
    rocrand_generator generator;
    ASSERT_EQ(
        rocrand_create_generator(
            &generator,
            ROCRAND_RNG_PSEUDO_PHILOX4_32_10
        ),
        ROCRAND_STATUS_SUCCESS
    );

    const size_t size = 256;
    double mean = 5.0;
    double stddev = 2.0;
    double * data;
    ASSERT_EQ(
        hipMalloc((void **)&data, size * sizeof(double)),
        hipSuccess
    );
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // n must be even
    EXPECT_EQ(
        rocrand_generate_normal_double(generator, (double *) data, 1, mean, stddev),
        ROCRAND_STATUS_LENGTH_NOT_MULTIPLE
    );

    // pointer must be aligned
    EXPECT_EQ(
        rocrand_generate_normal_double(generator, (double *)(data+1), 2, mean, stddev),
        ROCRAND_STATUS_LENGTH_NOT_MULTIPLE
    );

    EXPECT_EQ(
        rocrand_generate_normal_double(generator, (double *) data, size, mean, stddev),
        ROCRAND_STATUS_SUCCESS
    );

    EXPECT_EQ(hipFree(data), hipSuccess);
    EXPECT_EQ(
        rocrand_destroy_generator(generator),
        ROCRAND_STATUS_SUCCESS
    );
}

TEST(rocrand_generate_normal_tests, neg_test)
{
    const size_t size = 256;
    float mean = 5.0;
    float stddev = 2.0;
    float * data = NULL;

    rocrand_generator generator = NULL;
    EXPECT_EQ(
        rocrand_generate_normal(generator, (float *) data, size, mean, stddev),
        ROCRAND_STATUS_NOT_CREATED
    );
}
