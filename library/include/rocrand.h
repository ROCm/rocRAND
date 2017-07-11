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

#ifndef ROCRAND_H_
#define ROCRAND_H_

#include <hip/hip_runtime.h>

#ifndef ROCRANDAPI
#define ROCRANDAPI
#endif

#define ROCRAND_VERSION 100

/// rocRAND random number generator (opaque)
typedef struct rocrand_generator_base_type * rocrand_generator;

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * ROCRAND function call status types
 */
typedef enum rocrand_status {
    ROCRAND_STATUS_SUCCESS = 0, ///< No errors
    ROCRAND_STATUS_VERSION_MISMATCH = 100, ///< Header file and linked library version do not match
    ROCRAND_STATUS_NOT_INITIALIZED = 101, ///< Generator not initialized
    ROCRAND_STATUS_ALLOCATION_FAILED = 102, ///< Memory allocation failed
    ROCRAND_STATUS_TYPE_ERROR = 103, ///< Generator is wrong type
    ROCRAND_STATUS_OUT_OF_RANGE = 104, ///< Argument out of range
    ROCRAND_STATUS_LENGTH_NOT_MULTIPLE = 105, ///< Length requested is not a multple of dimension
    ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106, ///< GPU does not have double precision required by MRG32k3a
    ROCRAND_STATUS_LAUNCH_FAILURE = 201, ///< Kernel launch failure
    ROCRAND_STATUS_PREEXISTING_FAILURE = 202, ///< Preexisting failure on library entry
    ROCRAND_STATUS_INTERNAL_ERROR = 999 ///< Internal library error
} rocrand_status;

/**
 * ROCRAND generator types
 */
typedef enum rocrand_rng_type {
    ROCRAND_RNG_PSEUDO_XORWOW = 100, ///< XORWOW pseudorandom generator
    ROCRAND_RNG_PSEUDO_MRG32K3A = 200, ///< MRG32k3a pseudorandom generator
    ROCRAND_RNG_PSEUDO_MTGP32 = 300, ///< Mersenne Twister MTGP32 pseudorandom generator
    ROCRAND_RNG_PSEUDO_PHILOX4_32_10 = 400, ///< PHILOX-4x32-10 pseudorandom generator
    ROCRAND_RNG_QUASI_SOBOL32 = 500, ///< Sobol32 quasirandom generator
} rocrand_rng_type;


// Host API function

/**
 * \brief Create new random number generator.
 *
 * Creates a new random number generator of type \p rng_type
 * and returns it in \p *generator.
 *
 * Values for \p rng_type are:
 * - ROCRAND_RNG_PSEUDO_XORWOW
 * - ROCRAND_RNG_PSEUDO_MRG32K3A
 * - ROCRAND_RNG_PSEUDO_MTGP32
 * - ROCRAND_RNG_PSEUDO_PHILOX4_32_10
 * - ROCRAND_RNG_QUASI_SOBOL32
 *
 * \param generator - Pointer to generator
 * \param rng_type - Type of generator to create
 *
 * \return
 * - ROCRAND_STATUS_ALLOCATION_FAILED, if memory could not be allocated \n
 * - ROCRAND_STATUS_INITIALIZATION_FAILED if there was a problem setting up the GPU \n
 * - ROCRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
 *   dynamically linked library version \n
 * - ROCRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
 * - ROCRAND_STATUS_SUCCESS if generator was created successfully \n
 *
 */
rocrand_status ROCRANDAPI
rocrand_create_generator(rocrand_generator * generator, rocrand_rng_type rng_type);

/**
 * \brief Destroy generator.
 *
 * Destroy generator and free all memory (state)
 *
 * \param generator - Generator to destroy
 *
 * \return
 * - ROCRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - ROCRAND_STATUS_SUCCESS if generator was destroyed successfully \n
 */
rocrand_status ROCRANDAPI
rocrand_destroy_generator(rocrand_generator generator);

/**
 * \brief Generates uniformly distributed 32-bit unsigned integers.
 *
 * Generates \p n uniformly distributed 32-bit unsigned integers and
 * saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of 32-bit unsigned integers to generate
 *
 * \return
 * - ROCRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate(rocrand_generator generator,
                 unsigned int * output_data, size_t n);
    
/**
 * \brief Generates uniformly distributed floats.
 *
 * Generates \p n uniformly distributed floats and
 * saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of floats to generate
 *
 * \return
 * - ROCRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_uniform(rocrand_generator generator,
                         float * output_data, size_t n);

/**
 * \brief Generates normal distributed floats.
 *
 * Generates \p n normal distributed floats and
 * saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of floats to generate
 * \param mean - Mean value of normal distribution
 * \param stddev - Standard deviation value of normal distribution
 *
 * \return
 * - ROCRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_normal(rocrand_generator generator,
                        float * output_data, size_t n,
                        float mean, float stddev);

/**
 * \brief Generates normal distributed doubles.
 *
 * Generates \p n normal distributed doubles and
 * saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of doubles to generate
 * \param mean - Mean value of normal distribution
 * \param stddev - Standard deviation value of normal distribution
 *
 * \return
 * - ROCRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_normal_double(rocrand_generator generator,
                               double * output_data, size_t n,
                               double mean, double stddev);

/**
 * \brief Generates log-normal distributed floats.
 *
 * Generates \p n log-normal distributed floats and
 * saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of floats to generate
 * \param mean - Mean value of log normal distribution
 * \param stddev - Standard deviation value of log normal distribution
 *
 * \return
 * - ROCRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_log_normal(rocrand_generator generator,
                            float * output_data, size_t n,
                            float mean, float stddev);

/**
 * \brief Generates log-normal distributed doubles.
 *
 * Generates \p n log-normal distributed doubles and
 * saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of doubles to generate
 * \param mean - Mean value of log normal distribution
 * \param stddev - Standard deviation value of log normal distribution
 *
 * \return
 * - ROCRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_log_normal_double(rocrand_generator generator,
                                   double * output_data, size_t n,
                                   double mean, double stddev);

/**
 * \brief Generates Poisson-distributed 32-bit unsigned integers.
 *
 * Generates \p n Poisson-distributed 32-bit unsigned integers and
 * saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of 32-bit unsigned integers to generate
 * \param lambda - lambda for the Poisson distribution
 *
 * \return
 * - ROCRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if generator failed to launch kernel \n
 * - ROCRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_poisson(rocrand_generator generator,
                         unsigned int * output_data, size_t n,
                         double lambda);

/**
 * \brief Set the current stream for kernel launches.
 *
 * Set the current stream for all kernel launches of the generator.
 * All functions will use this stream.
 *
 * \param generator - Generator to modify
 * \param stream - Stream to use or NULL for null stream
 *
 * \return
 * - ROCRAND_STATUS_NOT_INITIALIZED if the generator was not initialized \n
 * - ROCRAND_STATUS_SUCCESS if stream was set successfully \n
 */
rocrand_status ROCRANDAPI
rocrand_set_stream(rocrand_generator generator, hipStream_t stream);

/**
 * \brief Return the version number of the library.
 *
 * Return in \p version the version number of the dynamically linked ROCRAND
 * library.
 *
 * \param version - Version of the library
 *
 * \return
 * - ROCRAND_STATUS_OUT_OF_RANGE if \p version is NULL \n
 * - ROCRAND_STATUS_SUCCESS if the version number was successfully returned \n
 */
rocrand_status ROCRANDAPI
rocrand_get_version(int * version);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // ROCRAND_H_
