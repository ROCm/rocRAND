// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

/** \rocrand_internal \addtogroup rocrandhost
 *
 *  @{
 */

#ifndef ROCRAND_H_
#define ROCRAND_H_

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include "rocrand_discrete_types.h"

/// \cond ROCRAND_DOCS_MACRO
#ifndef ROCRANDAPI
#ifdef WIN32
  #ifdef rocrand_EXPORTS
  /* We are building this library */
    #define ROCRANDAPI __declspec(dllexport)
  #else
    /* We are using this library */
    #define ROCRANDAPI __declspec(dllimport)
  #endif
#else
  #define ROCRANDAPI
#endif
#endif
/// \endcond

#include "rocrand_version.h"

/// \cond ROCRAND_DOCS_TYPEDEFS
/// rocRAND random number generator (opaque)
typedef struct rocrand_generator_base_type * rocrand_generator;
/// \endcond

/// \cond ROCRAND_DOCS_TYPEDEFS
/// rocRAND half type (derived from HIP)
typedef __half half;
/// \endcond

#define ROCRAND_DEFAULT_MAX_BLOCK_SIZE 256

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

/**
 * \brief rocRAND function call status type
 */
typedef enum rocrand_status {
    ROCRAND_STATUS_SUCCESS = 0, ///< No errors
    ROCRAND_STATUS_VERSION_MISMATCH = 100, ///< Header file and linked library version do not match
    ROCRAND_STATUS_NOT_CREATED = 101, ///< Generator was not created using rocrand_create_generator
    ROCRAND_STATUS_ALLOCATION_FAILED = 102, ///< Memory allocation failed during execution
    ROCRAND_STATUS_TYPE_ERROR = 103, ///< Generator type is wrong
    ROCRAND_STATUS_OUT_OF_RANGE = 104, ///< Argument out of range
    ROCRAND_STATUS_LENGTH_NOT_MULTIPLE = 105, ///< Requested size is not a multiple of quasirandom generator's dimension,
                                              ///< or requested size is not even (see rocrand_generate_normal()),
                                              ///< or pointer is misaligned (see rocrand_generate_normal())
    ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106, ///< GPU does not have double precision
    ROCRAND_STATUS_LAUNCH_FAILURE = 107, ///< Kernel launch failure
    ROCRAND_STATUS_INTERNAL_ERROR = 108 ///< Internal library error
} rocrand_status;

/**
 * \brief rocRAND generator type
 */
typedef enum rocrand_rng_type {
    ROCRAND_RNG_PSEUDO_DEFAULT = 400, ///< Default pseudorandom generator
    ROCRAND_RNG_PSEUDO_XORWOW = 401, ///< XORWOW pseudorandom generator
    ROCRAND_RNG_PSEUDO_MRG32K3A = 402, ///< MRG32k3a pseudorandom generator
    ROCRAND_RNG_PSEUDO_MTGP32 = 403, ///< Mersenne Twister MTGP32 pseudorandom generator
    ROCRAND_RNG_PSEUDO_PHILOX4_32_10 = 404, ///< PHILOX-4x32-10 pseudorandom generator
    ROCRAND_RNG_QUASI_DEFAULT = 500,  ///< Default quasirandom generator
    ROCRAND_RNG_QUASI_SOBOL32 = 501, ///< Sobol32 quasirandom generator
    ROCRAND_RNG_QUASI_SOBOL64 = 504 ///< Sobol64 quasirandom generator
} rocrand_rng_type;


// Host API function

/**
 * \brief Creates a new random number generator.
 *
 * Creates a new pseudo random number generator of type \p rng_type
 * and returns it in \p generator.
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
 * - ROCRAND_STATUS_VERSION_MISMATCH if the header file version does not match the
 *   dynamically linked library version \n
 * - ROCRAND_STATUS_TYPE_ERROR if the value for \p rng_type is invalid \n
 * - ROCRAND_STATUS_SUCCESS if generator was created successfully \n
 *
 */
rocrand_status ROCRANDAPI
rocrand_create_generator(rocrand_generator * generator, rocrand_rng_type rng_type);

/**
 * \brief Destroys random number generator.
 *
 * Destroys random number generator and frees related memory.
 *
 * \param generator - Generator to be destroyed
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
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
 * Generated numbers are between \p 0 and \p 2^32, including \p 0 and
 * excluding \p 2^32.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of 32-bit unsigned integers to generate
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
 * - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate(rocrand_generator generator,
                 unsigned int * output_data, size_t n);

/**
* \brief Generates uniformly distributed 8-bit unsigned integers.
*
* Generates \p n uniformly distributed 8-bit unsigned integers and
* saves them to \p output_data.
*
* Generated numbers are between \p 0 and \p 2^8, including \p 0 and
* excluding \p 2^8.
*
* \param generator - Generator to use
* \param output_data - Pointer to memory to store generated numbers
* \param n - Number of 8-bit unsigned integers to generate
*
* \return
* - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
* - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
* - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
* of used quasi-random generator \n
* - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
*/
rocrand_status ROCRANDAPI
rocrand_generate_char(rocrand_generator generator,
                      unsigned char * output_data, size_t n);

/**
* \brief Generates uniformly distributed 16-bit unsigned integers.
*
* Generates \p n uniformly distributed 16-bit unsigned integers and
* saves them to \p output_data.
*
* Generated numbers are between \p 0 and \p 2^16, including \p 0 and
* excluding \p 2^16.
*
* \param generator - Generator to use
* \param output_data - Pointer to memory to store generated numbers
* \param n - Number of 16-bit unsigned integers to generate
*
* \return
* - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
* - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
* - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
* of used quasi-random generator \n
* - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
*/
rocrand_status ROCRANDAPI
rocrand_generate_short(rocrand_generator generator,
                       unsigned short * output_data, size_t n);

/**
 * \brief Generates uniformly distributed \p float values.
 *
 * Generates \p n uniformly distributed 32-bit floating-point values
 * and saves them to \p output_data.
 *
 * Generated numbers are between \p 0.0f and \p 1.0f, excluding \p 0.0f and
 * including \p 1.0f.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of <tt>float</tt>s to generate
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
 * - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_uniform(rocrand_generator generator,
                         float * output_data, size_t n);

/**
 * \brief Generates uniformly distributed double-precision floating-point values.
 *
 * Generates \p n uniformly distributed 64-bit double-precision floating-point
 * values and saves them to \p output_data.
 *
 * Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
 * including \p 1.0.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of <tt>double</tt>s to generate
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
 * - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_uniform_double(rocrand_generator generator,
                                double * output_data, size_t n);

/**
 * \brief Generates uniformly distributed half-precision floating-point values.
 *
 * Generates \p n uniformly distributed 16-bit half-precision floating-point
 * values and saves them to \p output_data.
 *
 * Generated numbers are between \p 0.0 and \p 1.0, excluding \p 0.0 and
 * including \p 1.0.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of <tt>half</tt>s to generate
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
 * - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_uniform_half(rocrand_generator generator,
                              half * output_data, size_t n);

/**
 * \brief Generates normally distributed \p float values.
 *
 * Generates \p n normally distributed distributed 32-bit floating-point
 * values and saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of <tt>float</tt>s to generate
 * \param mean - Mean value of normal distribution
 * \param stddev - Standard deviation value of normal distribution
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
 * - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_normal(rocrand_generator generator,
                        float * output_data, size_t n,
                        float mean, float stddev);

/**
 * \brief Generates normally distributed \p double values.
 *
 * Generates \p n normally distributed 64-bit double-precision floating-point
 * numbers and saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of <tt>double</tt>s to generate
 * \param mean - Mean value of normal distribution
 * \param stddev - Standard deviation value of normal distribution
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
 * - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_normal_double(rocrand_generator generator,
                               double * output_data, size_t n,
                               double mean, double stddev);

/**
* \brief Generates normally distributed \p half values.
*
* Generates \p n normally distributed 16-bit half-precision floating-point
* numbers and saves them to \p output_data.
*
* \param generator - Generator to use
* \param output_data - Pointer to memory to store generated numbers
* \param n - Number of <tt>half</tt>s to generate
* \param mean - Mean value of normal distribution
* \param stddev - Standard deviation value of normal distribution
*
* \return
* - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
* - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
* - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
* of used quasi-random generator \n
* - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
*/
rocrand_status ROCRANDAPI
rocrand_generate_normal_half(rocrand_generator generator,
                             half * output_data, size_t n,
                             half mean, half stddev);

/**
 * \brief Generates log-normally distributed \p float values.
 *
 * Generates \p n log-normally distributed 32-bit floating-point values
 * and saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of <tt>float</tt>s to generate
 * \param mean - Mean value of log normal distribution
 * \param stddev - Standard deviation value of log normal distribution
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
 * - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_log_normal(rocrand_generator generator,
                            float * output_data, size_t n,
                            float mean, float stddev);

/**
 * \brief Generates log-normally distributed \p double values.
 *
 * Generates \p n log-normally distributed 64-bit double-precision floating-point
 * values and saves them to \p output_data.
 *
 * \param generator - Generator to use
 * \param output_data - Pointer to memory to store generated numbers
 * \param n - Number of <tt>double</tt>s to generate
 * \param mean - Mean value of log normal distribution
 * \param stddev - Standard deviation value of log normal distribution
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
 * - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_log_normal_double(rocrand_generator generator,
                                   double * output_data, size_t n,
                                   double mean, double stddev);

/**
* \brief Generates log-normally distributed \p half values.
*
* Generates \p n log-normally distributed 16-bit half-precision floating-point
* values and saves them to \p output_data.
*
* \param generator - Generator to use
* \param output_data - Pointer to memory to store generated numbers
* \param n - Number of <tt>half</tt>s to generate
* \param mean - Mean value of log normal distribution
* \param stddev - Standard deviation value of log normal distribution
*
* \return
* - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
* - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
* - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
* of used quasi-random generator \n
* - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
*/
rocrand_status ROCRANDAPI
rocrand_generate_log_normal_half(rocrand_generator generator,
                                 half * output_data, size_t n,
                                 half mean, half stddev);

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
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
 * - ROCRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
 * - ROCRAND_STATUS_LENGTH_NOT_MULTIPLE if \p n is not a multiple of the dimension
 * of used quasi-random generator \n
 * - ROCRAND_STATUS_SUCCESS if random numbers were successfully generated \n
 */
rocrand_status ROCRANDAPI
rocrand_generate_poisson(rocrand_generator generator,
                         unsigned int * output_data, size_t n,
                         double lambda);

/**
 * \brief Initializes the generator's state on GPU or host.
 *
 * Initializes the generator's state on GPU or host. User it not
 * required to call this function before using a generator.
 *
 * If rocrand_initialize() was not called for a generator, it will be
 * automatically called by functions which generates random numbers like
 * rocrand_generate(), rocrang_generate_uniform() etc.
 *
 * \param generator - Generator to initialize
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_LAUNCH_FAILURE if a HIP kernel launch failed \n
 * - ROCRAND_STATUS_SUCCESS if the seeds were generated successfully \n
 */
rocrand_status ROCRANDAPI
rocrand_initialize_generator(rocrand_generator generator);

/**
 * \brief Sets the current stream for kernel launches.
 *
 * Sets the current stream for all kernel launches of the generator.
 * All functions will use this stream.
 *
 * \param generator - Generator to modify
 * \param stream - Stream to use or NULL for default stream
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_SUCCESS if stream was set successfully \n
 */
rocrand_status ROCRANDAPI
rocrand_set_stream(rocrand_generator generator, hipStream_t stream);

/**
 * \brief Sets the seed of a pseudo-random number generator.
 *
 * Sets the seed of the pseudo-random number generator.
 *
 * - This operation resets the generator's internal state.
 * - This operation does not change the generator's offset.
 *
 * For a MRG32K3a generator seed value can't be zero. If \p seed is
 * equal zero and generator's type is ROCRAND_RNG_PSEUDO_MRG32K3A,
 * value \p 12345 is used as a seed instead.
 *
 * \param generator - Pseudo-random number generator
 * \param seed - New seed value
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_TYPE_ERROR if the generator is a quasi-random number generator \n
 * - ROCRAND_STATUS_SUCCESS if seed was set successfully \n
 */
rocrand_status ROCRANDAPI
rocrand_set_seed(rocrand_generator generator, unsigned long long seed);

/**
 * \brief Sets the offset of a random number generator.
 *
 * Sets the absolute offset of the random number generator.
 *
 * - This operation resets the generator's internal state.
 * - This operation does not change the generator's seed.
 *
 * Absolute offset cannot be set if generator's type is ROCRAND_RNG_PSEUDO_MTGP32.
 *
 * \param generator - Random number generator
 * \param offset - New absolute offset
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_SUCCESS if offset was successfully set \n
 * - ROCRAND_STATUS_TYPE_ERROR if generator's type is ROCRAND_RNG_PSEUDO_MTGP32
 */
rocrand_status ROCRANDAPI
rocrand_set_offset(rocrand_generator generator, unsigned long long offset);

/**
 * \brief Set the number of dimensions of a quasi-random number generator.
 *
 * Set the number of dimensions of a quasi-random number generator.
 * Supported values of \p dimensions are 1 to 20000.
 *
 * - This operation resets the generator's internal state.
 * - This operation does not change the generator's offset.
 *
 * \param generator - Quasi-random number generator
 * \param dimensions - Number of dimensions
 *
 * \return
 * - ROCRAND_STATUS_NOT_CREATED if the generator wasn't created \n
 * - ROCRAND_STATUS_TYPE_ERROR if the generator is not a quasi-random number generator \n
 * - ROCRAND_STATUS_OUT_OF_RANGE if \p dimensions is out of range \n
 * - ROCRAND_STATUS_SUCCESS if the number of dimensions was set successfully \n
 */
rocrand_status ROCRANDAPI
rocrand_set_quasi_random_generator_dimensions(rocrand_generator generator,
                                              unsigned int dimensions);

/**
 * \brief Returns the version number of the library.
 *
 * Returns in \p version the version number of the dynamically linked
 * rocRAND library.
 *
 * \param version - Version of the library
 *
 * \return
 * - ROCRAND_STATUS_OUT_OF_RANGE if \p version is NULL \n
 * - ROCRAND_STATUS_SUCCESS if the version number was successfully returned \n
 */
rocrand_status ROCRANDAPI
rocrand_get_version(int * version);

/**
 * \brief Construct the histogram for a Poisson distribution.
 *
 * Construct the histogram for the Poisson distribution with lambda \p lambda.
 *
 * \param lambda - lambda for the Poisson distribution
 * \param discrete_distribution - pointer to the histogram in device memory
 *
 * \return
 * - ROCRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null \n
 * - ROCRAND_STATUS_OUT_OF_RANGE if lambda is non-positive \n
 * - ROCRAND_STATUS_SUCCESS if the histogram was constructed successfully \n
 */
rocrand_status ROCRANDAPI
rocrand_create_poisson_distribution(double lambda,
                                    rocrand_discrete_distribution * discrete_distribution);

/**
 * \brief Construct the histogram for a custom discrete distribution.
 *
 * Construct the histogram for the discrete distribution of \p size
 * 32-bit unsigned integers from the range [\p offset, \p offset + \p size)
 * using \p probabilities as probabilities.
 *
 * \param probabilities - probabilities of the the distribution in host memory
 * \param size - size of \p probabilities
 * \param offset - offset of values
 * \param discrete_distribution - pointer to the histogram in device memory
 *
 * \return
 * - ROCRAND_STATUS_ALLOCATION_FAILED if memory could not be allocated \n
 * - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution pointer was null \n
 * - ROCRAND_STATUS_OUT_OF_RANGE if \p size was zero \n
 * - ROCRAND_STATUS_SUCCESS if the histogram was constructed successfully \n
 */
rocrand_status ROCRANDAPI
rocrand_create_discrete_distribution(const double * probabilities,
                                     unsigned int size,
                                     unsigned int offset,
                                     rocrand_discrete_distribution * discrete_distribution);

/**
 * \brief Destroy the histogram array for a discrete distribution.
 *
 * Destroy the histogram array for a discrete distribution created by
 * rocrand_create_poisson_distribution.
 *
 * \param discrete_distribution - pointer to the histogram in device memory
 *
 * \return
 * - ROCRAND_STATUS_OUT_OF_RANGE if \p discrete_distribution was null \n
 * - ROCRAND_STATUS_SUCCESS if the histogram was destroyed successfully \n
 */
rocrand_status ROCRANDAPI
rocrand_destroy_discrete_distribution(rocrand_discrete_distribution discrete_distribution);

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif // ROCRAND_H_

/** @} */ // end of group rocrandhost
