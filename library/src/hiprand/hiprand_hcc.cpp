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

#include <rocrand.h>

#include <hiprand.h>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

hiprandStatus_t to_hiprand_status(rocrand_status status)
{
    switch(status)
    {
        case ROCRAND_STATUS_SUCCESS:
            return HIPRAND_STATUS_SUCCESS;
        case ROCRAND_STATUS_NOT_CREATED:
            return HIPRAND_STATUS_NOT_INITIALIZED;
        case ROCRAND_STATUS_VERSION_MISMATCH:
            return HIPRAND_STATUS_VERSION_MISMATCH;
        case ROCRAND_STATUS_ALLOCATION_FAILED:
            return HIPRAND_STATUS_ALLOCATION_FAILED;
        case ROCRAND_STATUS_TYPE_ERROR:
            return HIPRAND_STATUS_TYPE_ERROR;
        case ROCRAND_STATUS_OUT_OF_RANGE:
            return HIPRAND_STATUS_OUT_OF_RANGE;
        case ROCRAND_STATUS_LENGTH_NOT_MULTIPLE:
            return HIPRAND_STATUS_LENGTH_NOT_MULTIPLE;
        case ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
            return HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED;
        case ROCRAND_STATUS_LAUNCH_FAILURE:
            return HIPRAND_STATUS_LAUNCH_FAILURE;
        // case ROCRAND_STATUS_PREEXISTING_FAILURE:
        //     return HIPRAND_STATUS_PREEXISTING_FAILURE;
        // case ROCRAND_STATUS_INITIALIZATION_FAILED:
        //     return HIPRAND_STATUS_INITIALIZATION_FAILED;
        // case ROCRAND_STATUS_ARCH_MISMATCH:
        //     return HIPRAND_STATUS_ARCH_MISMATCH;
        case ROCRAND_STATUS_INTERNAL_ERROR:
            return HIPRAND_STATUS_INTERNAL_ERROR;
        default:
            return HIPRAND_STATUS_INTERNAL_ERROR;
    }
}

rocrand_rng_type to_rocrand_rng_type(hiprandRngType_t rng_type)
{
    switch(rng_type)
    {
        case HIPRAND_RNG_PSEUDO_DEFAULT:
            return ROCRAND_RNG_PSEUDO_DEFAULT;
        case HIPRAND_RNG_PSEUDO_XORWOW:
            return ROCRAND_RNG_PSEUDO_XORWOW;
        case HIPRAND_RNG_PSEUDO_MRG32K3A:
            return ROCRAND_RNG_PSEUDO_MRG32K3A;
        case HIPRAND_RNG_PSEUDO_MTGP32:
            return ROCRAND_RNG_PSEUDO_MTGP32;
        case HIPRAND_RNG_PSEUDO_PHILOX4_32_10:
            return ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
        case HIPRAND_RNG_PSEUDO_MT19937:
            throw HIPRAND_STATUS_NOT_IMPLEMENTED;
        case HIPRAND_RNG_QUASI_DEFAULT:
            return ROCRAND_RNG_QUASI_DEFAULT;
        case HIPRAND_RNG_QUASI_SOBOL32:
            return ROCRAND_RNG_QUASI_SOBOL32;
        case HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32:
            throw HIPRAND_STATUS_NOT_IMPLEMENTED;
        case HIPRAND_RNG_QUASI_SOBOL64:
            throw HIPRAND_STATUS_NOT_IMPLEMENTED;
        case HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64:
            throw HIPRAND_STATUS_NOT_IMPLEMENTED;
        default:
            throw HIPRAND_STATUS_TYPE_ERROR;
    }
}

hiprandStatus_t HIPRANDAPI
hiprandCreateGenerator(hiprandGenerator_t * generator, hiprandRngType_t rng_type)
{
    try
    {
        return to_hiprand_status(
            rocrand_create_generator(
                (rocrand_generator *)(generator),
                to_rocrand_rng_type(rng_type)
            )
        );
    } catch(const hiprandStatus_t& error)
    {
        return error;
    }
}

hiprandStatus_t HIPRANDAPI
hiprandCreateGeneratorHost(hiprandGenerator_t * generator, hiprandRngType_t rng_type)
{
    (void) generator;
    (void) rng_type;
    return HIPRAND_STATUS_NOT_IMPLEMENTED;
}

hiprandStatus_t HIPRANDAPI
hiprandDestroyGenerator(hiprandGenerator_t generator)
{
    return to_hiprand_status(
        rocrand_destroy_generator(
            (rocrand_generator)(generator)
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerate(hiprandGenerator_t generator,
                unsigned int * output_data, size_t n)
{
    return to_hiprand_status(
        rocrand_generate(
            (rocrand_generator)(generator),
            output_data, n
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateUniform(hiprandGenerator_t generator,
                       float * output_data, size_t n)
{
    return to_hiprand_status(
        rocrand_generate_uniform(
            (rocrand_generator)(generator),
            output_data, n
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateUniformDouble(hiprandGenerator_t generator,
                             double * output_data, size_t n)
{
    return to_hiprand_status(
        rocrand_generate_uniform_double(
            (rocrand_generator)(generator),
            output_data, n
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateNormal(hiprandGenerator_t generator,
                      float * output_data, size_t n,
                      float mean, float stddev)
{
    return to_hiprand_status(
        rocrand_generate_normal(
            (rocrand_generator)(generator),
            output_data, n,
            mean, stddev
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateNormalDouble(hiprandGenerator_t generator,
                            double * output_data, size_t n,
                            double mean, double stddev)
{
    return to_hiprand_status(
        rocrand_generate_normal_double(
            (rocrand_generator)(generator),
            output_data, n,
            mean, stddev
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateLogNormal(hiprandGenerator_t generator,
                         float * output_data, size_t n,
                         float mean, float stddev)
{
    return to_hiprand_status(
        rocrand_generate_log_normal(
            (rocrand_generator)(generator),
            output_data, n,
            mean, stddev
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateLogNormalDouble(hiprandGenerator_t generator,
                               double * output_data, size_t n,
                               double mean, double stddev)
{
    return to_hiprand_status(
        rocrand_generate_log_normal_double(
            (rocrand_generator)(generator),
            output_data, n,
            mean, stddev
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGeneratePoisson(hiprandGenerator_t generator,
                       unsigned int * output_data, size_t n,
                       double lambda)
{
    return to_hiprand_status(
        rocrand_generate_poisson(
            (rocrand_generator)(generator),
            output_data, n,
            lambda
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGenerateSeeds(hiprandGenerator_t generator)
{
    return to_hiprand_status(
        rocrand_initialize_generator(
            (rocrand_generator)(generator)
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandSetStream(hiprandGenerator_t generator, hipStream_t stream)
{
    return to_hiprand_status(
        rocrand_set_stream(
            (rocrand_generator)(generator),
            stream
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandSetPseudoRandomGeneratorSeed(hiprandGenerator_t generator, unsigned long long seed)
{
    return to_hiprand_status(
        rocrand_set_seed(
            (rocrand_generator)(generator),
            seed
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandSetGeneratorOffset(hiprandGenerator_t generator, unsigned long long offset)
{
    return to_hiprand_status(
        rocrand_set_offset(
            (rocrand_generator)(generator),
            offset
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandSetQuasiRandomGeneratorDimensions(hiprandGenerator_t generator, unsigned int dimensions)
{
    return to_hiprand_status(
        rocrand_set_quasi_random_generator_dimensions(
            (rocrand_generator)(generator),
            dimensions
        )
    );
}

hiprandStatus_t HIPRANDAPI
hiprandGetVersion(int * version)
{
    return to_hiprand_status(
        rocrand_get_version(version)
    );
}

hiprandStatus_t HIPRANDAPI
hiprandCreatePoissonDistribution(double lambda, hiprandDiscreteDistribution_t * discrete_distribution)
{
    return to_hiprand_status(
        rocrand_create_poisson_distribution(lambda, discrete_distribution)
    );
}

hiprandStatus_t HIPRANDAPI
hiprandDestroyDistribution(hiprandDiscreteDistribution_t discrete_distribution)
{
    return to_hiprand_status(
        rocrand_destroy_discrete_distribution(discrete_distribution)
    );
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */
