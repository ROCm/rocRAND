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

#ifndef ROCRAND_HPP_
#define ROCRAND_HPP_

// At least C++11 required
#if defined(__cplusplus) && __cplusplus >= 201103L

#include <random>
#include <exception>
#include <string>
#include <sstream>
#include <type_traits>
#include <limits>

#include "rocrand.h"
#include "rocrand_kernel.h"

namespace rocrand_cpp {

/// \addtogroup rocrandhostcpp
/// @{

/// \class error
/// \brief A run-time rocRAND error.
///
/// The error class represents an error returned
/// by a rocRAND function.
class error : public std::exception
{
public:
    /// rocRAND error code type
    typedef rocrand_status error_type;

    /// Constructs new error object from error code \p error.
    ///
    /// \param error - error code
    error(error_type error) noexcept
        : m_error(error),
          m_error_string(to_string(error))
    {
    }

    ~error() noexcept
    {
    }

    /// Returns the numeric error code.
    error_type error_code() const noexcept
    {
        return m_error;
    }

    /// Returns a string description of the error.
    std::string error_string() const noexcept
    {
        return m_error_string;
    }

    /// Returns a C-string description of the error.
    const char* what() const noexcept
    {
        return m_error_string.c_str();
    }

    /// Static function which converts the numeric rocRAND
    /// error code \p error to a human-readable string.
    ///
    /// If the error code is unknown, a string containing
    /// "Unknown rocRAND error" along with the error code
    /// \p error will be returned.
    static std::string to_string(error_type error)
    {
        switch(error)
        {
            case ROCRAND_STATUS_SUCCESS:
                return "Success";
            case ROCRAND_STATUS_VERSION_MISMATCH:
                return "Header file and linked library version do not match";
            case ROCRAND_STATUS_NOT_CREATED:
                return "Generator was not created using rocrand_create_generator";
            case ROCRAND_STATUS_ALLOCATION_FAILED:
                return "Memory allocation failed during execution";
            case ROCRAND_STATUS_TYPE_ERROR:
                return "Generator type is wrong";
            case ROCRAND_STATUS_OUT_OF_RANGE:
                return "Argument out of range";
            case ROCRAND_STATUS_LENGTH_NOT_MULTIPLE:
                return "Length requested is not a multple of dimension";
            case ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                return "GPU does not have double precision";
            case ROCRAND_STATUS_LAUNCH_FAILURE:
                return "Kernel launch failure";
            case ROCRAND_STATUS_INTERNAL_ERROR:
                return "Internal library error";
            default: {
                std::stringstream s;
                s << "Unknown rocRAND error (" << error << ")";
                return s.str();
            }
        }
    }

    /// Compares two error objects for equality.
    friend
    bool operator==(const error& l, const error& r)
    {
        return l.error_code() == r.error_code();
    }

    /// Compares two error objects for inequality.
    friend
    bool operator!=(const error& l, const error& r)
    {
        return !(l == r);
    }

private:
    error_type m_error;
    std::string m_error_string;
};

/// \class uniform_int_distribution
///
/// \brief Produces random integer values uniformly distributed on the interval [0, 2^32 - 1].
///
/// \tparam IntType - type of generated values. Only \p unsigned \p int type is supported.
template<class IntType = unsigned int>
class uniform_int_distribution
{
    static_assert(
        std::is_same<unsigned int, IntType>::value,
            "Only unsigned int type is supported in uniform_int_distribution"
    );

public:
    typedef IntType result_type;

    /// Default constructor
    uniform_int_distribution()
    {
    }

    /// Resets distribution's internal state if there is any.
    void reset()
    {
    }

    /// Returns the smallest possible value that can be generated.
    IntType min() const
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated.
    IntType max() const
    {
        return std::numeric_limits<IntType>::max();
    }

    /// \brief Fills \p output with uniformly distributed random integer values.
    ///
    /// Generates \p size random integer values uniformly distributed
    /// on the  interval [0, 2^32 - 1], and stores them into the device memory
    /// referenced by \p output pointer.
    ///
    /// \param g - An uniform random number generator object
    /// \param output - Pointer to device memory to store results
    /// \param size - Number of values to generate
    ///
    /// The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p IntType type.
    ///
    /// See also: rocrand_generate()
    template<class Generator>
    void operator()(Generator& g, IntType * output, size_t size)
    {
        rocrand_status status = rocrand_generate(g.m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// Returns \c true if the distribution is the same as \p other.
    bool operator==(const uniform_int_distribution<IntType>& other)
    {
        return true;
    }

    /// Returns \c true if the distribution is different from \p other.
    bool operator!=(const uniform_int_distribution<IntType>& other)
    {
        return !(*this == other);
    }
};

/// \class uniform_real_distribution
///
/// \brief Produces random floating-point values uniformly distributed on the interval (0, 1].
///
/// \tparam RealType - type of generated values. Only \p float and \p double types are supported.
template<class RealType = float>
class uniform_real_distribution
{
    static_assert(
            std::is_same<float, RealType>::value
            || std::is_same<double, RealType>::value,
            "Only float and double types are supported in uniform_real_distribution"
    );

public:
    typedef RealType result_type;

    /// Default constructor
    uniform_real_distribution()
    {
    }

    /// Resets distribution's internal state if there is any.
    void reset()
    {
    }

    /// Returns the smallest possible value that can be generated.
    RealType min() const
    {
        return std::nextafter(RealType(0.0), RealType(1.0));
    }

    /// Returns the largest possible value that can be generated.
    RealType max() const
    {
        return 1.0;
    }

    /// \brief Fills \p output with uniformly distributed random floating-point values.
    ///
    /// Generates \p size random floating-point values uniformly distributed
    /// on the interval (0, 1], and stores them into the device memory referenced
    /// by \p output pointer.
    ///
    /// \param g - An uniform random number generator object
    /// \param output - Pointer to device memory to store results
    /// \param size - Number of values to generate
    ///
    /// The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p RealType type.
    ///
    /// See also: rocrand_generate_uniform(), rocrand_generate_uniform_double()
    template<class Generator>
    void operator()(Generator& g, RealType * output, size_t size)
    {
        rocrand_status status;
        status = this->generate(g, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// Returns \c true if the distribution is the same as \p other.
    bool operator==(const uniform_real_distribution<RealType>& other)
    {
        return true;
    }

    /// Returns \c true if the distribution is different from \p other.
    bool operator!=(const uniform_real_distribution<RealType>& other)
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    rocrand_status generate(Generator& g, float * output, size_t size)
    {
        return rocrand_generate_uniform(g.m_generator, output, size);
    }

    template<class Generator>
    rocrand_status generate(Generator& g, double * output, size_t size)
    {
        return rocrand_generate_uniform_double(g.m_generator, output, size);
    }
};

/// \class normal_distribution
///
/// \brief Produces random numbers according to a normal distribution.
///
/// \tparam RealType - type of generated values. Only \p float and \p double types are supported.
///
/// See also: <a href="https://en.wikipedia.org/wiki/Normal_distribution">Wikipedia:Normal distribution</a>.
template<class RealType = float>
class normal_distribution
{
    static_assert(
            std::is_same<float, RealType>::value
            || std::is_same<double, RealType>::value,
            "Only float and double types are supported in normal_distribution"
    );

public:
    typedef RealType result_type;

    /// \class param_type
    /// \brief The type of the distribution parameter set.
    class param_type
    {
    public:
        using distribution_type = normal_distribution<RealType>;
        param_type(RealType mean = 0.0, RealType stddev = 1.0)
            : m_mean(mean), m_stddev(stddev)
        {
        }

        param_type(const param_type& params)
        : m_mean(params.mean()), m_stddev(params.stddev())
        {
        }

        /// \brief Returns the deviation distribution parameter.
        ///
        /// The default value is 1.0.
        RealType mean() const
        {
            return m_mean;
        }

        /// \brief Returns the standard deviation distribution parameter.
        ///
        /// The default value is 1.0.
        RealType stddev() const
        {
            return m_stddev;
        }

        /// Returns \c true if the param_type is the same as \p other.
        bool operator==(const param_type& other)
        {
            return m_mean == other.m_mean && m_stddev == other.m_stddev;
        }

        /// Returns \c true if the param_type is different from \p other.
        bool operator!=(const param_type& other)
        {
            return !(*this == other);
        }
    private:
        RealType m_mean;
        RealType m_stddev;
    };

    /// \brief Constructs a new distribution object.
    /// \param mean - A mean distribution parameter
    /// \param stddev - A standard deviation distribution parameter
    normal_distribution(RealType mean = 0.0, RealType stddev = 1.0)
        : m_params(mean, stddev)
    {
    }

    /// \brief Constructs a new distribution object.
    /// \param params - Distribution parameters
    normal_distribution(const param_type& params)
        : m_params(params)
    {
    }

    /// Resets distribution's internal state if there is any.
    void reset()
    {
    }

    /// \brief Returns the mean distribution parameter.
    ///
    /// The mean specifies the location of the peak. The default value is 0.0.
    RealType mean() const
    {
        return m_params.mean();
    }

    /// \brief Returns the standard deviation distribution parameter.
    ///
    /// The default value is 1.0.
    RealType stddev() const
    {
        return m_params.stddev();
    }

    /// Returns the smallest possible value that can be generated.
    RealType min() const
    {
        return std::numeric_limits<RealType>::lowest();
    }

    /// Returns the largest possible value that can be generated.
    RealType max() const
    {
        return std::numeric_limits<RealType>::max();
    }

    /// Returns the distribution parameter object
    param_type param() const
    {
        return m_params;
    }

    /// Sets the distribution parameter object
    void param(const param_type& params)
    {
        m_params = params;
    }

    /// \brief Fills \p output with normally distributed random floating-point values.
    ///
    /// Generates \p size random floating-point values distributed according to a normal distribution,
    /// and stores them into the device memory referenced by \p output pointer.
    ///
    /// \param g - An uniform random number generator object
    /// \param output - Pointer to device memory to store results
    /// \param size - Number of values to generate
    ///
    /// The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p RealType type.
    ///
    /// See also: rocrand_generate_normal(), rocrand_generate_normal_double()
    template<class Generator>
    void operator()(Generator& g, RealType * output, size_t size)
    {
        rocrand_status status;
        status = this->generate(g, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \brief Returns \c true if the distribution is the same as \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator==(const normal_distribution<RealType>& other)
    {
        return this->m_params == other.m_params;
    }

    /// \brief Returns \c true if the distribution is different from \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator!=(const normal_distribution<RealType>& other)
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    rocrand_status generate(Generator& g, float * output, size_t size)
    {
        return rocrand_generate_normal(
            g.m_generator, output, size, this->mean(), this->stddev()
        );
    }

    template<class Generator>
    rocrand_status generate(Generator& g, double * output, size_t size)
    {
        return rocrand_generate_normal_double(
            g.m_generator, output, size, this->mean(), this->stddev()
        );
    }

    param_type m_params;
};

/// \class lognormal_distribution
///
/// \brief Produces positive random numbers according to a log-normal distribution.
///
/// \tparam RealType - type of generated values. Only \p float and \p double types are supported.
///
/// See also: <a href="https://en.wikipedia.org/wiki/Log-normal_distribution">Wikipedia:Log-normal distribution</a>.
template<class RealType = float>
class lognormal_distribution
{
    static_assert(
            std::is_same<float, RealType>::value
            || std::is_same<double, RealType>::value,
            "Only float and double types are supported in lognormal_distribution"
    );

public:
    typedef RealType result_type;

    /// \class param_type
    /// \brief The type of the distribution parameter set.
    class param_type
    {
    public:
        using distribution_type = lognormal_distribution<RealType>;
        param_type(RealType m = 0.0, RealType s = 1.0)
            : m_mean(m), m_stddev(s)
        {
        }

        param_type(const param_type& params)
        : m_mean(params.m()), m_stddev(params.s())
        {
        }

        /// \brief Returns the deviation distribution parameter.
        ///
        /// The default value is 1.0.
        RealType m() const
        {
            return m_mean;
        }

        /// \brief Returns the deviation distribution parameter.
        ///
        /// The default value is 1.0.
        RealType s() const
        {
            return m_stddev;
        }

        /// Returns \c true if the param_type is the same as \p other.
        bool operator==(const param_type& other)
        {
            return m_mean == other.m_mean && m_stddev == other.m_stddev;
        }

        /// Returns \c true if the param_type is different from \p other.
        bool operator!=(const param_type& other)
        {
            return !(*this == other);
        }
    private:
        RealType m_mean;
        RealType m_stddev;
    };

    /// \brief Constructs a new distribution object.
    /// \param m - A mean distribution parameter
    /// \param s - A standard deviation distribution parameter
    lognormal_distribution(RealType m = 0.0, RealType s = 1.0)
        : m_params(m, s)
    {
    }

    /// \brief Constructs a new distribution object.
    /// \param params - Distribution parameters
    lognormal_distribution(const param_type& params)
        : m_params(params)
    {
    }

    /// Resets distribution's internal state if there is any.
    void reset()
    {
    }

    /// \brief Returns the mean distribution parameter.
    ///
    /// The mean specifies the location of the peak. The default value is 0.0.
    RealType m() const
    {
        return m_params.m();
    }

    /// \brief Returns the standard deviation distribution parameter.
    ///
    /// The default value is 1.0.
    RealType s() const
    {
        return m_params.s();
    }

    /// Returns the distribution parameter object
    param_type param() const
    {
        return m_params;
    }

    /// Sets the distribution parameter object
    void param(const param_type& params)
    {
        m_params = params;
    }

    /// Returns the smallest possible value that can be generated.
    RealType min() const
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated.
    RealType max() const
    {
        return std::numeric_limits<RealType>::max();
    }

    /// \brief Fills \p output with log-normally distributed random floating-point values.
    ///
    /// Generates \p size random floating-point values (greater than zero) distributed according
    /// to a log-normal distribution, and stores them into the device memory referenced
    /// by \p output pointer.
    ///
    /// \param g - An uniform random number generator object
    /// \param output - Pointer to device memory to store results
    /// \param size - Number of values to generate
    ///
    /// The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p RealType type.
    ///
    /// See also: rocrand_generate_log_normal(), rocrand_generate_log_normal_double()
    template<class Generator>
    void operator()(Generator& g, RealType * output, size_t size)
    {
        rocrand_status status;
        status = this->generate(g, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \brief Returns \c true if the distribution is the same as \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator==(const lognormal_distribution<RealType>& other)
    {
        return this->m_params == other.m_params;
    }

    /// \brief Returns \c true if the distribution is different from \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator!=(const lognormal_distribution<RealType>& other)
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    rocrand_status generate(Generator& g, float * output, size_t size)
    {
        return rocrand_generate_log_normal(
            g.m_generator, output, size, this->m(), this->s()
        );
    }

    template<class Generator>
    rocrand_status generate(Generator& g, double * output, size_t size)
    {
        return rocrand_generate_log_normal_double(
            g.m_generator, output, size, this->m(), this->s()
        );
    }

    param_type m_params;
};

/// \class poisson_distribution
///
/// \brief Produces random non-negative integer values distributed according to Poisson distribution.
///
/// \tparam IntType - type of generated values. Only \p unsinged \p int type is supported.
///
/// See also: <a href="https://en.wikipedia.org/wiki/Poisson_distribution">Wikipedia:Poisson distribution</a>.
template<class IntType = unsigned int>
class poisson_distribution
{
    static_assert(
        std::is_same<unsigned int, IntType>::value,
        "Only unsigned int type is supported in poisson_distribution"
    );

public:
    typedef IntType result_type;

    /// \class param_type
    /// \brief The type of the distribution parameter set.
    class param_type
    {
    public:
        using distribution_type = poisson_distribution<IntType>;
        param_type(double mean = 1.0)
            : m_mean(mean)
        {
        }

        param_type(const param_type& params)
        : m_mean(params.mean())
        {
        }

        /// \brief Returns the mean distribution parameter.
        ///
        /// The mean (also known as lambda) is the average number
        /// of events per interval. The default value is 1.0.
        double mean() const
        {
            return m_mean;
        }

        /// Returns \c true if the param_type is the same as \p other.
        bool operator==(const param_type& other)
        {
            return m_mean == other.m_mean;
        }

        /// Returns \c true if the param_type is different from \p other.
        bool operator!=(const param_type& other)
        {
            return !(*this == other);
        }

    private:
        double m_mean;
    };

    /// \brief Constructs a new distribution object.
    /// \param mean - A mean distribution parameter.
    poisson_distribution(double mean = 1.0)
        : m_params(mean)
    {
    }

    /// \brief Constructs a new distribution object.
    /// \param params - Distribution parameters
    poisson_distribution(const param_type& params)
        : m_params(params)
    {
    }

    /// Resets distribution's internal state if there is any.
    void reset()
    {
    }

    /// \brief Returns the mean distribution parameter.
    ///
    /// The mean (also known as lambda) is the average number
    /// of events per interval. The default value is 1.0.
    double mean() const
    {
        return m_params.mean();
    }

    /// Returns the smallest possible value that can be generated.
    IntType min() const
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated.
    IntType max() const
    {
        return std::numeric_limits<IntType>::max();
    }

    /// Returns the distribution parameter object
    param_type param() const
    {
        return m_params;
    }

    /// Sets the distribution parameter object
    void param(const param_type& params)
    {
        m_params = params;
    }

    /// \brief Fills \p output with random non-negative integer values
    /// distributed according to Poisson distribution.
    ///
    /// Generates \p size random non-negative integer values distributed according
    /// to Poisson distribution, and stores them into the device memory referenced
    /// by \p output pointer.
    ///
    /// \param g - An uniform random number generator object
    /// \param output - Pointer to device memory to store results
    /// \param size - Number of values to generate
    ///
    /// The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p IntType type.
    ///
    /// See also: rocrand_generate_poisson()
    template<class Generator>
    void operator()(Generator& g, IntType * output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate_poisson(g.m_generator, output, size, this->mean());
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \brief Returns \c true if the distribution is the same as \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator==(const poisson_distribution<IntType>& other)
    {
        return this->m_params == other.m_params;
    }

    /// \brief Returns \c true if the distribution is different from \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator!=(const poisson_distribution<IntType>& other)
    {
        return !(*this == other);
    }

private:
    param_type m_params;
};

namespace detail {

template<rocrand_rng_type GeneratorType>
class rng_engine
{
public:
    /// \typedef result_type
    /// Type of values generated by the random number generator.
    typedef unsigned int result_type;
    /// \typedef offset_type
    /// Random number generator offset type.
    /// Offset represents a number of RNG's states that should be skipped
    /// before first value is generated.
    ///
    /// See also: offset()
    typedef unsigned long long offset_type;

    rng_engine(offset_type offset_value)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, GeneratorType);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
        if(offset_value > 0)
        {
            this->offset(offset_value);
        }
    }

    rng_engine(rocrand_generator& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    ~rng_engine()
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \brief Sets RNG's \p hipStream for kernel launches.
    /// \param value - new \p hipStream to use
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \brief Sets the offset of a random number generator.
    ///
    /// Offset represents a number of RNG's states that should be skipped
    /// before first value is generated.
    ///
    /// - This operation resets the engine's internal state.
    /// - This operation does not change the engine's seed or the number of dimensions.
    ///
    /// \param value - New absolute offset
    ///
    /// See also: rocrand_set_offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \brief Fills \p output with uniformly distributed random integer values.
    ///
    /// Generates \p size random integer values uniformly distributed
    /// on the  interval [0, 2^32 - 1], and stores them into the device memory
    /// referenced by \p output pointer.
    ///
    /// \param output - Pointer to device memory to store results
    /// \param size - Number of values to generate
    ///
    /// The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p IntType type.
    ///
    /// See also: rocrand_generate()
    template<class Generator>
    void operator()(IntType * output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// Returns the smallest possible value that can be generated by the engine.
    result_type min() const
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated by the engine.
    result_type max() const
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// Returns RNG type.
    static constexpr rocrand_rng_type type()
    {
        return GeneratorType;
    }

protected:
    rocrand_generator m_generator;

    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
};

template<rocrand_rng_type GeneratorType, unsigned long long DefaultSeed>
class prng_engine : public rng_engine<GeneratorType>
{
    typedef rng_engine<GeneratorType> base_type;

public:
    /// \typedef seed_type
    /// Pseudo-random number generator seed type definition.
    typedef unsigned long long seed_type; // 64bit uint

    /// \brief The default seed equal to \p DefaultSeed.
    static constexpr seed_type default_seed = DefaultSeed;

    prng_engine(seed_type seed_value = DefaultSeed,
                typename base_type::offset_type offset_value = 0)
        : base_type(offset_value)
    {
        this->seed(seed_value);
    }

    prng_engine(rocrand_generator& generator)
        : base_type(generator)
    {
    }

    ~prng_engine()
    {
    }

    /// \brief Sets the seed of the pseudo-random number generator.
    ///
    /// - This operation resets the generator's internal state.
    /// - This operation does not change the generator's offset.
    ///
    /// \param value - New seed value
    ///
    /// See also: rocrand_set_seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }
};

template<rocrand_rng_type GeneratorType, unsigned long long DefaultSeed>
constexpr typename prng_engine<GeneratorType, DefaultSeed>::seed_type
prng_engine<GeneratorType, DefaultSeed>::default_seed;

template<rocrand_rng_type GeneratorType, unsigned int DefaultNumDimensions = 1>
class qrng_engine : public rng_engine<GeneratorType>
{
    typedef rng_engine<GeneratorType> base_type;

public:
    /// \typedef dimensions_num_type
    /// Quasi-random number generator seed type definition.
    ///
    /// See also dimensions()
    typedef unsigned int dimensions_num_type; // 32-bit uint

    /// \brief The default number of dimenstions, equal to \p DefaultNumDimensions.
    static constexpr dimensions_num_type default_num_dimensions = DefaultNumDimensions;

    qrng_engine(dimensions_num_type num_of_dimensions = DefaultNumDimensions,
                typename base_type::offset_type offset_value = 0)
        : base_type(offset_value)
    {
        this->dimensions(num_of_dimensions);
    }

    qrng_engine(rocrand_generator& generator)
        : base_type(generator)
    {
    }

    ~qrng_engine()
    {
    }

    /// \brief Set the number of dimensions of a quasi-random number generator.
    ///
    /// Supported values of \p dimensions are 1 to 20000.
    ///
    /// - This operation resets the generator's internal state.
    /// - This operation does not change the generator's offset.
    ///
    /// \param value - Number of dimensions
    ///
    /// See also: rocrand_set_quasi_random_generator_dimensions()
    void dimensions(dimensions_num_type value)
    {
        rocrand_status status =
            rocrand_set_quasi_random_generator_dimensions(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }
};

template<rocrand_rng_type GeneratorType, unsigned int DefaultNumDimensions>
constexpr typename qrng_engine<GeneratorType, DefaultNumDimensions>::dimensions_num_type
qrng_engine<GeneratorType, DefaultNumDimensions>::default_num_dimensions;

} // end detail namespace

template<unsigned long long DefaultSeed = ROCRAND_PHILOX4x32_DEFAULT_SEED>
class philox4x32_10_engine : public detail::prng_engine<ROCRAND_RNG_PSEUDO_PHILOX4_32_10, DefaultSeed>
{
    typedef detail::prng_engine<ROCRAND_RNG_PSEUDO_PHILOX4_32_10, DefaultSeed> base_type;
public:
    using base_type::base_type;
};

template<unsigned long long DefaultSeed = ROCRAND_XORWOW_DEFAULT_SEED>
class xorwow_engine : public detail::prng_engine<ROCRAND_RNG_PSEUDO_XORWOW, DefaultSeed>
{
    typedef detail::prng_engine<ROCRAND_RNG_PSEUDO_XORWOW, DefaultSeed> base_type;
public:
    using base_type::base_type;
};

template<unsigned long long DefaultSeed = ROCRAND_MRG32K3A_DEFAULT_SEED>
class mrg32k3a_engine : public detail::prng_engine<ROCRAND_RNG_PSEUDO_MRG32K3A, DefaultSeed>
{
    typedef detail::prng_engine<ROCRAND_RNG_PSEUDO_MRG32K3A, DefaultSeed> base_type;
public:
    using base_type::base_type;
};

template<unsigned long long DefaultSeed = 0>
class mtgp32_engine : public detail::prng_engine<ROCRAND_RNG_PSEUDO_MTGP32, DefaultSeed>
{
    typedef detail::prng_engine<ROCRAND_RNG_PSEUDO_MTGP32, DefaultSeed> base_type;
public:
    using base_type::base_type;
};

template<unsigned int DefaultNumDimensions = 1>
class sobol32_engine : public detail::qrng_engine<ROCRAND_RNG_QUASI_SOBOL32, DefaultNumDimensions>
{
    typedef detail::qrng_engine<ROCRAND_RNG_QUASI_SOBOL32, DefaultNumDimensions> base_type;
public:
    using base_type::base_type;

    /// \typedef dimensions_num_type
    /// Quasi-random number generator seed type definition.
    ///
    /// See also dimensions()
    typedef unsigned int dimensions_num_type; // 32-bit uint
};

/// \typedef philox4x32_10;
/// \brief Typedef of rocrand_cpp::philox4x32_10_engine PRNG engine with default seed (#ROCRAND_PHILOX4x32_DEFAULT_SEED).
typedef philox4x32_10_engine<> philox4x32_10;
/// \typedef xorwow
/// \brief Typedef of rocrand_cpp::xorwow_engine PRNG engine with default seed (#ROCRAND_XORWOW_DEFAULT_SEED).
typedef xorwow_engine<> xorwow;
/// \typedef mrg32k3a
/// \brief Typedef of rocrand_cpp::mrg32k3a_engine PRNG engine with default seed (#ROCRAND_MRG32K3A_DEFAULT_SEED).
typedef mrg32k3a_engine<> mrg32k3a;
/// \typedef mtgp32
/// \brief Typedef of rocrand_cpp::mtgp32_engine PRNG engine with default seed (0).
typedef mtgp32_engine<> mtgp32;
/// \typedef sobol32
/// \brief Typedef of rocrand_cpp::sobol32_engine PRNG engine with default number of dimensions (1).
typedef sobol32_engine<> sobol32;

/// \typedef default_random_engine
/// \brief Default random engine.
typedef xorwow default_random_engine;

/// \typedef random_device
///
/// \brief A non-deterministic uniform random number generator,
/// see <a href="http://en.cppreference.com/w/cpp/numeric/random/random_device">std::random_device</a>.
///
/// rocrand_cpp::random_device is non-deterministic uniform random number generator,
/// or a pseudo-random number engine if there is no support for non-deterministic
/// random number generation. It's implemented as a typedef of
/// <a href="http://en.cppreference.com/w/cpp/numeric/random/random_device">std::random_device</a>.
///
/// For practical use rocrand_cpp::random_device is generally only used to seed a PRNG
/// such as \ref rocrand_cpp::mtgp32_engine.
///
/// Example:
/// \code
/// #include <rocrand.hpp>
///
/// int main()
/// {
///     const size_t size = 8192;
///     unsigned int * output;
///     hipMalloc(&output, size * sizeof(unsigned int));
///
///     rocrand_cpp::random_device rd;
///     rocrand_cpp::mtgp32 engine(rd()); // seed engine with a real random value, if available
///     rocrand_cpp::normal_distribution<float> dist(0.0, 1.5);
///     dist(engine, output, size);
/// }
/// \endcode
typedef std::random_device random_device;

/// \brief Returns rocRAND version.
/// \return rocRAND version number as an \p int value.
int version()
{
    int x;
    rocrand_status status = rocrand_get_version(&x);
    if(status != ROCRAND_STATUS_SUCCESS)
    {
        throw rocrand_cpp::error(status);
    }
    return x;
}

/// @} // end of group rocrandhostcpp

} // end namespace rocrand_cpp

#endif // #if __cplusplus >= 201103L
#endif // ROCRAND_HPP_
