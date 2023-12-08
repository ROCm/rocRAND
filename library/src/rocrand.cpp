// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rng/generator_type.hpp"
#include "rng/generator_types.hpp"

#include <new>
#include <rocrand/rocrand.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

rocrand_status ROCRANDAPI rocrand_create_generator(rocrand_generator* generator,
                                                   rocrand_rng_type   rng_type)
{
    try
    {
        // clang-format off
        switch(rng_type)
        {
            case ROCRAND_RNG_PSEUDO_PHILOX4_32_10:
                *generator = new rocrand_generator_type<rocrand_philox4x32_10>();
                break;
            case ROCRAND_RNG_PSEUDO_MRG31K3P:
                *generator = new rocrand_generator_type<rocrand_mrg31k3p>();
                break;
            case ROCRAND_RNG_PSEUDO_MRG32K3A:
                *generator = new rocrand_generator_type<rocrand_mrg32k3a>();
                break;
            case ROCRAND_RNG_PSEUDO_DEFAULT:
            case ROCRAND_RNG_PSEUDO_XORWOW:
                *generator = new rocrand_generator_type<rocrand_xorwow>();
                break;
            case ROCRAND_RNG_QUASI_DEFAULT:
            case ROCRAND_RNG_QUASI_SOBOL32:
                *generator = new rocrand_generator_type<rocrand_sobol32>();
                break;
            case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32:
                *generator = new rocrand_generator_type<rocrand_scrambled_sobol32>();
                break;
            case ROCRAND_RNG_QUASI_SOBOL64:
                *generator = new rocrand_generator_type<rocrand_sobol64>();
                break;
            case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64:
                *generator = new rocrand_generator_type<rocrand_scrambled_sobol64>();
                break;
            case ROCRAND_RNG_PSEUDO_MTGP32:
                *generator = new rocrand_generator_type<rocrand_mtgp32>();
                break;
            case ROCRAND_RNG_PSEUDO_LFSR113:
                *generator = new rocrand_generator_type<rocrand_lfsr113>();
                break;
            case ROCRAND_RNG_PSEUDO_MT19937:
                *generator = new rocrand_generator_type<rocrand_mt19937>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY2_32_20:
                *generator = new rocrand_generator_type<rocrand_threefry2x32_20>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY2_64_20:
                *generator = new rocrand_generator_type<rocrand_threefry2x64_20>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY4_32_20:
                *generator = new rocrand_generator_type<rocrand_threefry4x32_20>();
                break;
            case ROCRAND_RNG_PSEUDO_THREEFRY4_64_20:
                *generator = new rocrand_generator_type<rocrand_threefry4x64_20>();
                break;
            default:
                return ROCRAND_STATUS_TYPE_ERROR;
        }
        // clang-format on
    }
    catch(const std::bad_alloc& e)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }
    catch(rocrand_status status)
    {
        return status;
    }
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_create_generator_host(rocrand_generator* generator,
                                                        rocrand_rng_type   rng_type)
{
    try
    {
        // clang-format off
        switch(rng_type)
        {
            case ROCRAND_RNG_PSEUDO_LFSR113:
                *generator = new rocrand_generator_type<rocrand_lfsr113_host>();
                break;
            case ROCRAND_RNG_PSEUDO_PHILOX4_32_10:
                *generator = new rocrand_generator_type<rocrand_philox4x32_10_host>();
                break;
            case ROCRAND_RNG_PSEUDO_MRG31K3P:
                *generator = new rocrand_generator_type<rocrand_mrg31k3p_host>();
                break;
            case ROCRAND_RNG_QUASI_DEFAULT:
            case ROCRAND_RNG_QUASI_SOBOL32:
                *generator = new rocrand_generator_type<rocrand_sobol32_host>();
                break;
            case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32:
                *generator = new rocrand_generator_type<rocrand_scrambled_sobol32_host>();
                break;
            case ROCRAND_RNG_QUASI_SOBOL64:
                *generator = new rocrand_generator_type<rocrand_sobol64_host>();
                break;
            case ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64:
                *generator = new rocrand_generator_type<rocrand_scrambled_sobol64_host>();
                break;
            case ROCRAND_RNG_PSEUDO_DEFAULT:
            case ROCRAND_RNG_PSEUDO_XORWOW:
                *generator = new rocrand_generator_type<rocrand_xorwow_host>();
                break;
            case ROCRAND_RNG_PSEUDO_MRG32K3A:
            case ROCRAND_RNG_PSEUDO_MTGP32:
            case ROCRAND_RNG_PSEUDO_MT19937:
            case ROCRAND_RNG_PSEUDO_THREEFRY2_32_20:
            case ROCRAND_RNG_PSEUDO_THREEFRY2_64_20:
            case ROCRAND_RNG_PSEUDO_THREEFRY4_32_20:
            case ROCRAND_RNG_PSEUDO_THREEFRY4_64_20:
            default:
                return ROCRAND_STATUS_TYPE_ERROR;
        }
        // clang-format on
    }
    catch(const std::bad_alloc& e)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }
    catch(rocrand_status status)
    {
        return status;
    }
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_destroy_generator(rocrand_generator generator)
{
    try
    {
        delete generator;
    }
    catch(rocrand_status status)
    {
        return status;
    }
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_generate(rocrand_generator generator,
                                           unsigned int*     output_data,
                                           size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_int(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_long_long(rocrand_generator       generator,
                                                     unsigned long long int* output_data,
                                                     size_t                  n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_long(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_char(rocrand_generator generator,
                                                unsigned char*    output_data,
                                                size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_char(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_short(rocrand_generator generator,
                                                 unsigned short*   output_data,
                                                 size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_short(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_uniform(rocrand_generator generator,
                                                   float*            output_data,
                                                   size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_uniform_float(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_uniform_double(rocrand_generator generator,
                                                          double*           output_data,
                                                          size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_uniform_double(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_uniform_half(rocrand_generator generator,
                                                        half*             output_data,
                                                        size_t            n)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_uniform_half(output_data, n);
}

rocrand_status ROCRANDAPI rocrand_generate_normal(
    rocrand_generator generator, float* output_data, size_t n, float mean, float stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_normal_float(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_normal_double(
    rocrand_generator generator, double* output_data, size_t n, double mean, double stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_normal_double(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_normal_half(
    rocrand_generator generator, half* output_data, size_t n, half mean, half stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_normal_half(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_log_normal(
    rocrand_generator generator, float* output_data, size_t n, float mean, float stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_log_normal_float(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_log_normal_double(
    rocrand_generator generator, double* output_data, size_t n, double mean, double stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_log_normal_double(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_log_normal_half(
    rocrand_generator generator, half* output_data, size_t n, half mean, half stddev)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->generate_log_normal_half(output_data, n, mean, stddev);
}

rocrand_status ROCRANDAPI rocrand_generate_poisson(rocrand_generator generator,
                                                   unsigned int*     output_data,
                                                   size_t            n,
                                                   double            lambda)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }
    if(lambda <= 0.0)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    return generator->generate_poisson(output_data, n, lambda);
}

rocrand_status ROCRANDAPI rocrand_initialize_generator(rocrand_generator generator)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->init();
}

rocrand_status ROCRANDAPI rocrand_set_stream(rocrand_generator generator, hipStream_t stream)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    generator->set_stream(stream);
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_set_seed(rocrand_generator generator, unsigned long long seed)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    generator->set_seed(seed);
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_set_seed_uint4(rocrand_generator generator, uint4 seed)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->set_seed_uint4(seed);
}

rocrand_status ROCRANDAPI rocrand_set_offset(rocrand_generator generator, unsigned long long offset)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->set_offset(offset);
}

rocrand_status ROCRANDAPI rocrand_set_ordering(rocrand_generator generator, rocrand_ordering order)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->set_order(order);
}

rocrand_status ROCRANDAPI rocrand_set_quasi_random_generator_dimensions(rocrand_generator generator,
                                                                        unsigned int dimensions)
{
    if(generator == NULL)
    {
        return ROCRAND_STATUS_NOT_CREATED;
    }

    return generator->set_dimensions(dimensions);
}

rocrand_status ROCRANDAPI rocrand_get_version(int* version)
{
    if(version == NULL)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    *version = ROCRAND_VERSION;
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_create_poisson_distribution(
    double lambda, rocrand_discrete_distribution* discrete_distribution)
{
    if(discrete_distribution == NULL)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }
    if(lambda <= 0.0)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    rocrand_poisson_distribution<ROCRAND_DISCRETE_METHOD_UNIVERSAL> h_dis;
    try
    {
        h_dis = rocrand_poisson_distribution<ROCRAND_DISCRETE_METHOD_UNIVERSAL>(lambda);
    }
    catch(const std::exception& e)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }
    catch(rocrand_status status)
    {
        return status;
    }

    hipError_t error;
    error = hipMalloc(discrete_distribution, sizeof(rocrand_discrete_distribution_st));
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_ALLOCATION_FAILED;
    }
    error = hipMemcpy(*discrete_distribution,
                      &h_dis,
                      sizeof(rocrand_discrete_distribution_st),
                      hipMemcpyDefault);
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI
    rocrand_create_discrete_distribution(const double*                  probabilities,
                                         unsigned int                   size,
                                         unsigned int                   offset,
                                         rocrand_discrete_distribution* discrete_distribution)
{
    if(discrete_distribution == NULL)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }
    if(size == 0)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    rocrand_discrete_distribution_base<ROCRAND_DISCRETE_METHOD_UNIVERSAL> h_dis;
    try
    {
        h_dis = rocrand_discrete_distribution_base<ROCRAND_DISCRETE_METHOD_UNIVERSAL>(probabilities,
                                                                                      size,
                                                                                      offset);
    }
    catch(const std::exception& e)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }
    catch(rocrand_status status)
    {
        return status;
    }

    hipError_t error;
    error = hipMalloc(discrete_distribution, sizeof(rocrand_discrete_distribution_st));
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_ALLOCATION_FAILED;
    }
    error = hipMemcpy(*discrete_distribution,
                      &h_dis,
                      sizeof(rocrand_discrete_distribution_st),
                      hipMemcpyDefault);
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI
    rocrand_destroy_discrete_distribution(rocrand_discrete_distribution discrete_distribution)
{
    if(discrete_distribution == NULL)
    {
        return ROCRAND_STATUS_OUT_OF_RANGE;
    }

    rocrand_discrete_distribution_base<ROCRAND_DISCRETE_METHOD_UNIVERSAL> h_dis;

    hipError_t error;
    error = hipMemcpy(&h_dis,
                      discrete_distribution,
                      sizeof(rocrand_discrete_distribution_st),
                      hipMemcpyDefault);
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    try
    {
        h_dis.deallocate();
    }
    catch(rocrand_status status)
    {
        return status;
    }

    error = hipFree(discrete_distribution);
    if(error != hipSuccess)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }

    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_get_direction_vectors32(const unsigned int**         vectors,
                                                          rocrand_direction_vector_set set)
{
    switch(set)
    {
        case ROCRAND_DIRECTION_VECTORS_32_JOEKUO6:
            *vectors = rocrand_h_sobol32_direction_vectors;
            return ROCRAND_STATUS_SUCCESS;
        case ROCRAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6:
            *vectors = rocrand_h_scrambled_sobol32_direction_vectors;
            return ROCRAND_STATUS_SUCCESS;
        default: return ROCRAND_STATUS_OUT_OF_RANGE;
    }
}

rocrand_status ROCRANDAPI rocrand_get_direction_vectors64(const unsigned long long**   vectors,
                                                          rocrand_direction_vector_set set)
{
    switch(set)
    {
        case ROCRAND_DIRECTION_VECTORS_64_JOEKUO6:
            *vectors = rocrand_h_sobol64_direction_vectors;
            return ROCRAND_STATUS_SUCCESS;
        case ROCRAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6:
            *vectors = rocrand_h_scrambled_sobol64_direction_vectors;
            return ROCRAND_STATUS_SUCCESS;
        default: return ROCRAND_STATUS_OUT_OF_RANGE;
    }
}

rocrand_status ROCRANDAPI rocrand_get_scramble_constants32(const unsigned int** constants)
{
    *constants = h_scrambled_sobol32_constants;
    return ROCRAND_STATUS_SUCCESS;
}

rocrand_status ROCRANDAPI rocrand_get_scramble_constants64(const unsigned long long** constants)
{
    *constants = h_scrambled_sobol64_constants;
    return ROCRAND_STATUS_SUCCESS;
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */
