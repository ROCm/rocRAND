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

#include <hip/hip_runtime.h>

#include "rng/generators.hpp"

#include <rocrand.h>
#include <new>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

rocrand_status ROCRANDAPI
rocrand_create_generator(rocrand_generator * generator, rocrand_rng_type rng_type)
{
    try
    {
        if(rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
        {
            *generator = new rocrand_philox4x32_10();
        }
        else if(rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
        {
            *generator = new rocrand_mrg32k3a();
        }
        else if(rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
        {
            *generator = new rocrand_xorwow();
        }
        else
        {
            return ROCRAND_STATUS_TYPE_ERROR;
        }
    }
    catch(const std::bad_alloc& e)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }
    catch(rocrand_status status)
    {
        return ROCRAND_STATUS_SUCCESS;
    }
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI
rocrand_destroy_generator(rocrand_generator generator)
{
    try
    {
        delete(generator);
    }
    catch(rocrand_status status)
    {
        return status;
    }
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI
rocrand_generate(rocrand_generator generator,
                 unsigned int * output_data, size_t n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        rocrand_philox4x32_10 * philox4x32_10_generator =
            static_cast<rocrand_philox4x32_10 *>(generator);
        return philox4x32_10_generator->generate(output_data, n);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        rocrand_mrg32k3a * mrg32k3a_generator =
            static_cast<rocrand_mrg32k3a *>(generator);
        return mrg32k3a_generator->generate(output_data, n);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        rocrand_xorwow * rocrand_xorwow_generator =
            static_cast<rocrand_xorwow *>(generator);
        return rocrand_xorwow_generator->generate(output_data, n);
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_generate_uniform(rocrand_generator generator,
                         float * output_data, size_t n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        rocrand_philox4x32_10 * philox4x32_10_generator =
            static_cast<rocrand_philox4x32_10 *>(generator);
        return philox4x32_10_generator->generate_uniform(output_data, n);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        rocrand_mrg32k3a * mrg32k3a_generator =
            static_cast<rocrand_mrg32k3a *>(generator);
        return mrg32k3a_generator->generate_uniform(output_data, n);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        rocrand_xorwow * rocrand_xorwow_generator =
            static_cast<rocrand_xorwow *>(generator);
        return rocrand_xorwow_generator->generate_uniform(output_data, n);
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_generate_uniform_double(rocrand_generator generator,
                                double * output_data, size_t n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        rocrand_philox4x32_10 * philox4x32_10_generator =
            static_cast<rocrand_philox4x32_10 *>(generator);
        return philox4x32_10_generator->generate_uniform(output_data, n);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        rocrand_mrg32k3a * mrg32k3a_generator =
            static_cast<rocrand_mrg32k3a *>(generator);
        return mrg32k3a_generator->generate_uniform(output_data, n);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        rocrand_xorwow * rocrand_xorwow_generator =
            static_cast<rocrand_xorwow *>(generator);
        return rocrand_xorwow_generator->generate_uniform(output_data, n);
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_generate_normal(rocrand_generator generator,
                        float * output_data, size_t n,
                        float mean, float stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        rocrand_philox4x32_10 * philox4x32_10_generator =
            static_cast<rocrand_philox4x32_10 *>(generator);
        return philox4x32_10_generator->generate_normal(output_data, n,
                                                        stddev, mean);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        rocrand_mrg32k3a * mrg32k3a_generator =
            static_cast<rocrand_mrg32k3a *>(generator);
        return mrg32k3a_generator->generate_normal(output_data, n,
                                                   stddev, mean);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        rocrand_xorwow * rocrand_xorwow_generator =
            static_cast<rocrand_xorwow *>(generator);
        return rocrand_xorwow_generator->generate_normal(output_data, n,
                                                         stddev, mean);
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_generate_normal_double(rocrand_generator generator,
                               double * output_data, size_t n,
                               double mean, double stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        rocrand_philox4x32_10 * philox4x32_10_generator =
            static_cast<rocrand_philox4x32_10 *>(generator);
        return philox4x32_10_generator->generate_normal(output_data, n,
                                                        stddev, mean);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        rocrand_mrg32k3a * mrg32k3a_generator =
            static_cast<rocrand_mrg32k3a *>(generator);
        return mrg32k3a_generator->generate_normal(output_data, n,
                                                   stddev, mean);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        rocrand_xorwow * rocrand_xorwow_generator =
            static_cast<rocrand_xorwow *>(generator);
        return rocrand_xorwow_generator->generate_normal(output_data, n,
                                                         stddev, mean);
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_generate_log_normal(rocrand_generator generator,
                            float * output_data, size_t n,
                            float mean, float stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        rocrand_philox4x32_10 * philox4x32_10_generator =
            static_cast<rocrand_philox4x32_10 *>(generator);
        return philox4x32_10_generator->generate_log_normal(output_data, n,
                                                            stddev, mean);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        rocrand_mrg32k3a * mrg32k3a_generator =
            static_cast<rocrand_mrg32k3a *>(generator);
        return mrg32k3a_generator->generate_log_normal(output_data, n,
                                                       stddev, mean);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        rocrand_xorwow * rocrand_xorwow_generator =
            static_cast<rocrand_xorwow *>(generator);
        return rocrand_xorwow_generator->generate_log_normal(output_data, n,
                                                             stddev, mean);
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_generate_log_normal_double(rocrand_generator generator,
                                   double * output_data, size_t n,
                                   double mean, double stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        rocrand_philox4x32_10 * philox4x32_10_generator =
            static_cast<rocrand_philox4x32_10 *>(generator);
        return philox4x32_10_generator->generate_log_normal(output_data, n,
                                                            stddev, mean);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        rocrand_mrg32k3a * mrg32k3a_generator =
            static_cast<rocrand_mrg32k3a *>(generator);
        return mrg32k3a_generator->generate_log_normal(output_data, n,
                                                       stddev, mean);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        rocrand_xorwow * rocrand_xorwow_generator =
            static_cast<rocrand_xorwow *>(generator);
        return rocrand_xorwow_generator->generate_log_normal(output_data, n,
                                                             stddev, mean);
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_generate_poisson(rocrand_generator generator,
                         unsigned int * output_data, size_t n,
                         double lambda)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }
    if (lambda <= 0.0)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        rocrand_philox4x32_10 * philox4x32_10_generator =
            static_cast<rocrand_philox4x32_10 *>(generator);
        return philox4x32_10_generator->generate_poisson(output_data, n,
                                                         lambda);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        rocrand_mrg32k3a * mrg32k3a_generator =
            static_cast<rocrand_mrg32k3a *>(generator);
        return mrg32k3a_generator->generate_poisson(output_data, n,
                                                    lambda);
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        rocrand_xorwow * rocrand_xorwow_generator =
            static_cast<rocrand_xorwow *>(generator);
        return rocrand_xorwow_generator->generate_poisson(output_data, n,
                                                          lambda);
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_initialize_generator(rocrand_generator generator)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        return static_cast<rocrand_philox4x32_10 *>(generator)->init();
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        return static_cast<rocrand_mrg32k3a *>(generator)->init();
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        return static_cast<rocrand_xorwow *>(generator)->init();
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_set_stream(rocrand_generator generator, hipStream_t stream)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        static_cast<rocrand_philox4x32_10 *>(generator)->set_stream(stream);
        return ROCRAND_STATUS_SUCCESS;
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        static_cast<rocrand_mrg32k3a *>(generator)->set_stream(stream);
        return ROCRAND_STATUS_SUCCESS;
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        static_cast<rocrand_xorwow*>(generator)->set_stream(stream);
        return ROCRAND_STATUS_SUCCESS;
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_set_seed(rocrand_generator generator, unsigned long long seed)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        static_cast<rocrand_philox4x32_10 *>(generator)->set_seed(seed);
        return ROCRAND_STATUS_SUCCESS;
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        if(seed == 0ULL)
        {
            seed = ROCRAND_MRG32K3A_DEFAULT_SEED;
        }
        static_cast<rocrand_mrg32k3a *>(generator)->set_seed(seed);
        return ROCRAND_STATUS_SUCCESS;
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        static_cast<rocrand_xorwow *>(generator)->set_seed(seed);
        return ROCRAND_STATUS_SUCCESS;
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_set_offset(rocrand_generator generator, unsigned long long offset)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    if(generator->rng_type == ROCRAND_RNG_PSEUDO_PHILOX4_32_10)
    {
        static_cast<rocrand_philox4x32_10 *>(generator)->set_offset(offset);
        return ROCRAND_STATUS_SUCCESS;
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_MRG32K3A)
    {
        static_cast<rocrand_mrg32k3a *>(generator)->set_offset(offset);
        return ROCRAND_STATUS_SUCCESS;
    }
    else if(generator->rng_type == ROCRAND_RNG_PSEUDO_XORWOW)
    {
        static_cast<rocrand_xorwow *>(generator)->set_offset(offset);
        return ROCRAND_STATUS_SUCCESS;
    }
    return ROCRAND_STATUS_TYPE_ERROR;
}

rocrand_status ROCRANDAPI
rocrand_get_version(int * version)
{
    if(version == NULL)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    *version = ROCRAND_VERSION;
    return ROCRAND_STATUS_SUCCESS;
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */
