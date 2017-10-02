# Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

## @addtogroup rocrandpython
# @{

## @namespace rocrand.rocrand
# rocRAND wrapper

## @}

import os
import ctypes
import ctypes.util
from ctypes import *

import numbers
import numpy as np

from .hip import load_hip, HIP_PATHS
from .hip import empty, device_pointer

from .utils import find_library, expand_paths

## @cond INCLUDE_INTERNAL

rocrand = None

ROCRAND_PATHS = [
        os.getenv("ROCRAND_PATH")
    ] + expand_paths(HIP_PATHS, ["", "rocrand"])

def load_rocrand():
    global rocrand

    # Try to load libraries to resolve all dependencies
    # (e.g. libhip_hcc.so and its dependencies)
    try:
        CDLL(find_library(ROCRAND_PATHS, "librocrand.so"))
    except OSError as e:
        pass

    load_hip()

    try:
        rocrand = CDLL(find_library(ROCRAND_PATHS, "librocrand.so"))
    except OSError as e:
        raise ImportError("librocrand.so cannot be loaded: " + str(e))

load_rocrand()

ROCRAND_RNG_PSEUDO_DEFAULT = 400
ROCRAND_RNG_PSEUDO_XORWOW = 401
ROCRAND_RNG_PSEUDO_MRG32K3A = 402
ROCRAND_RNG_PSEUDO_MTGP32 = 403
ROCRAND_RNG_PSEUDO_PHILOX4_32_10 = 404
ROCRAND_RNG_QUASI_DEFAULT = 500
ROCRAND_RNG_QUASI_SOBOL32 = 501

ROCRAND_STATUS_SUCCESS = 0
ROCRAND_STATUS_VERSION_MISMATCH = 100
ROCRAND_STATUS_NOT_CREATED = 101
ROCRAND_STATUS_ALLOCATION_FAILED = 102
ROCRAND_STATUS_TYPE_ERROR = 103
ROCRAND_STATUS_OUT_OF_RANGE = 104
ROCRAND_STATUS_LENGTH_NOT_MULTIPLE = 105
ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106
ROCRAND_STATUS_LAUNCH_FAILURE = 107
ROCRAND_STATUS_INTERNAL_ERROR = 108

ROCRAND_STATUS = {
    ROCRAND_STATUS_SUCCESS: (
        "ROCRAND_STATUS_SUCCESS",
        "Success"),
    ROCRAND_STATUS_VERSION_MISMATCH: (
        "ROCRAND_STATUS_VERSION_MISMATCH",
        "Header file and linked library version do not match"),
    ROCRAND_STATUS_NOT_CREATED: (
        "ROCRAND_STATUS_NOT_CREATED",
        "Generator was not created using rocrand_create_generator"),
    ROCRAND_STATUS_ALLOCATION_FAILED: (
        "ROCRAND_STATUS_ALLOCATION_FAILED",
        "Memory allocation failed during execution"),
    ROCRAND_STATUS_TYPE_ERROR: (
        "ROCRAND_STATUS_TYPE_ERROR",
        "Generator type is wrong"),
    ROCRAND_STATUS_OUT_OF_RANGE: (
        "ROCRAND_STATUS_OUT_OF_RANGE",
        "Argument out of range"),
    ROCRAND_STATUS_LENGTH_NOT_MULTIPLE: (
        "ROCRAND_STATUS_LENGTH_NOT_MULTIPLE",
        "Length requested is not a multiple of dimension"),
    ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED: (
        "ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED",
        "GPU does not have double precision"),
    ROCRAND_STATUS_LAUNCH_FAILURE: (
        "ROCRAND_STATUS_LAUNCH_FAILURE",
        "Kernel launch failure"),
    ROCRAND_STATUS_INTERNAL_ERROR: (
        "ROCRAND_STATUS_INTERNAL_ERROR",
        "Internal library error")
}

def check_rocrand(status):
    if status != ROCRAND_STATUS_SUCCESS:
        raise RocRandError(status)

## @endcond # INCLUDE_INTERNAL

## Run-time rocRAND error.
class RocRandError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        if self.value in ROCRAND_STATUS:
            v, s = ROCRAND_STATUS[self.value]
        else:
            v, s = str(self.value), "Unknown error"
        return "{} ({})".format(s, v)


## Random number generator base class.
class RNG(object):
    ## @property offset
    # Mutatable attribute of the offset of random numbers sequence.
    #
    # Setting this attribute resets the sequence.

    ## @property stream
    # Mutatable attribute of HIP stream for all kernel launches of the generator.
    #
    # All functions will use this stream.
    # @c None means default stream.

    def __init__(self, rngtype, offset=None, stream=None):
        self._gen = c_void_p()
        check_rocrand(rocrand.rocrand_create_generator(byref(self._gen), rngtype))

        self._offset = 0
        if offset is not None:
            self.offset = offset

        self._stream = None
        if stream is not None:
            self.stream = stream

    ## @cond INCLUDE_INTERNAL

    def __del__(self):
        if self._gen:
            check_rocrand(rocrand.rocrand_destroy_generator(self._gen))

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offset):
        check_rocrand(rocrand.rocrand_set_offset(self._gen, c_ulonglong(offset)))
        self._offset = offset

    @property
    def stream(self):
        return self._stream

    @stream.setter
    def stream(self, stream):
        check_rocrand(rocrand.rocrand_set_stream(self._gen, stream))
        self._stream = stream

    ## @endcond # INCLUDE_INTERNAL

    def _generate(self, gen_func, ary, size, *args):
        if size is not None:
            if size > ary.size:
                raise ValueError("requested size is greater than ary")
        else:
            size = ary.size

        if isinstance(ary, np.ndarray):
            dary, needs_conversion = empty(size, ary.dtype), True
        elif isinstance(ary, DeviceNDArray):
            dary, needs_conversion = ary, False
        else:
            raise TypeError("unsupported type {}".format(type(ary)))

        check_rocrand(gen_func(self._gen, device_pointer(dary), c_size_t(size), *args))

        if needs_conversion:
            dary.copy_to_host(ary)

    ## Generates uniformly distributed integers.
    #
    # Generates @p size (if present) or @p ary.size uniformly distributed
    # integers and saves them to @p ary.
    #
    # Supported @c dtype of @p ary: @c numpy.uint32, @c numpy.int32.
    #
    # @param ary  NumPy array (@c numpy.ndarray) or
    #   HIP device-side array (@c DeviceNDArray)
    # @param size Number of samples to generate, default to @p ary.size
    #
    def generate(self, ary, size=None):
        if ary.dtype in (np.uint32, np.int32):
            self._generate(
                rocrand.rocrand_generate,
                ary, size)
        else:
            raise TypeError("unsupported type {}".format(ary.dtype))

    ## Generates uniformly distributed floats.
    #
    # Generates @p size (if present) or @p ary.size uniformly distributed
    # floats and saves them to @p ary.
    #
    # Supported @c dtype of @p ary: @c numpy.float32, @c numpy.float64.
    #
    # Generated numbers are between \c 0.0 and \c 1.0, excluding \c 0.0 and
    # including \c 1.0.
    #
    # @param ary  NumPy array (@c numpy.ndarray) or
    #   HIP device-side array (@c DeviceNDArray)
    # @param size Number of samples to generate, default to @p ary.size
    #
    def uniform(self, ary, size=None):
        if ary.dtype == np.float32:
            self._generate(
                rocrand.rocrand_generate_uniform,
                ary, size)
        elif ary.dtype == np.float64:
            self._generate(
                rocrand.rocrand_generate_uniform_double,
                ary, size)
        else:
            raise TypeError("unsupported type {}".format(ary.dtype))

    ## Generates normally distributed floats.
    #
    # Generates @p size (if present) or @p ary.size normally distributed
    # floats and saves them to @p ary.
    #
    # Supported @c dtype of @p ary: @c numpy.float32, @c numpy.float64.
    #
    # @param ary    NumPy array (@c numpy.ndarray) or
    #   HIP device-side array (@c DeviceNDArray)
    # @param mean   Mean value of normal distribution
    # @param stddev Standard deviation value of normal distribution
    # @param size   Number of samples to generate, default to @p ary.size
    #
    def normal(self, ary, mean, stddev, size=None):
        if ary.dtype == np.float32:
            self._generate(
                rocrand.rocrand_generate_normal,
                ary, size,
                c_float(mean), c_float(stddev))
        elif ary.dtype == np.float64:
            self._generate(
                rocrand.rocrand_generate_normal_double,
                ary, size,
                c_double(mean), c_double(stddev))
        else:
            raise TypeError("unsupported type {}".format(ary.dtype))

    ## Generates log-normally distributed floats.
    #
    # Generates @p size (if present) or @p ary.size log-normally distributed
    # floats and saves them to @p ary.
    #
    # Supported @c dtype of @p ary: @c numpy.float32, @c numpy.float64.
    #
    # @param ary    NumPy array (@c numpy.ndarray) or
    #   HIP device-side array (@c DeviceNDArray)
    # @param mean   Mean value of log normal distribution
    # @param stddev Standard deviation value of log normal distribution
    # @param size   Number of samples to generate, default to @p ary.size
    #
    def lognormal(self, ary, mean, stddev, size=None):
        if ary.dtype == np.float32:
            self._generate(
                rocrand.rocrand_generate_log_normal,
                ary, size,
                c_float(mean), c_float(stddev))
        elif ary.dtype == np.float64:
            self._generate(
                rocrand.rocrand_generate_log_normal_double,
                ary, size,
                c_double(mean), c_double(stddev))
        else:
            raise TypeError("unsupported type {}".format(ary.dtype))

    ## Generates Poisson-distributed integers.
    #
    # Generates @p size (if present) or @p ary.size Poisson-distributed
    # integers and saves them to @p ary.
    #
    # Supported @c dtype of @p ary: @c numpy.uint32, @c numpy.int32.
    #
    # @param ary   NumPy array (@c numpy.ndarray) or
    #   HIP device-side array (@c DeviceNDArray)
    # @param lmbd  lambda for the Poisson distribution
    # @param size  Number of samples to generate, default to @p ary.size
    #
    def poisson(self, ary, lmbd, size=None):
        if ary.dtype in (np.uint32, np.int32):
            self._generate(
                rocrand.rocrand_generate_poisson,
                ary, size,
                c_double(lmbd))
        else:
            raise TypeError("unsupported type {}".format(ary.dtype))


## Pseudo-random number generator.
#
# Example:
#
# @code{.py}
#
#   import rocrand
#   import numpy as np
#
#   gen = rocrand.PRNG(rocrand.PRNG.PHILOX4_32_10, seed=123456)
#   a = np.empty(1000, np.int32)
#   gen.poisson(a, 10.0)
#   print(a)
#
# @endcode
class PRNG(RNG):
    ## Default pseudo-random generator type, @ref XORWOW
    DEFAULT       = ROCRAND_RNG_PSEUDO_DEFAULT
    ## XORWOW pseudo-random generator type
    XORWOW        = ROCRAND_RNG_PSEUDO_XORWOW
    ## MRG32k3a pseudo-random generator type
    MRG32K3A      = ROCRAND_RNG_PSEUDO_MRG32K3A
    ## Mersenne Twister MTGP32 pseudo-random generator type
    MTGP32        = ROCRAND_RNG_PSEUDO_MTGP32
    ## PHILOX_4x32 (10 rounds) pseudo-random generator type
    PHILOX4_32_10 = ROCRAND_RNG_PSEUDO_PHILOX4_32_10

    ## @property seed
    # Mutatable attribute of the seed of random numbers sequence.
    #
    # Setting this attribute resets the sequence.

    ## @brief Creates a new pseudo-random number generator.
    #
    # A new pseudo-random number generator of type @p rngtype is initialized
    # with given @p seed, @p offset and @p stream.
    #
    # Values of @p rngtype:
    # - @ref DEFAULT
    # - @ref XORWOW
    # - @ref MRG32K3A
    # - @ref MTGP32
    # - @ref PHILOX4_32_10
    #
    # @param rngtype Type of pseudo-random number generator to create
    # @param seed    Initial seed value
    # @param offset  Initial offset of random numbers sequence
    # @param stream  HIP stream for all kernel launches of the generator
    #
    def __init__(self, rngtype=DEFAULT, seed=None, offset=None, stream=None):
        super(PRNG, self).__init__(rngtype, offset=offset, stream=stream)

        self._seed = None
        if seed is not None:
            self.seed = seed

    ## @cond INCLUDE_INTERNAL

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, seed):
        check_rocrand(rocrand.rocrand_set_seed(self._gen, c_ulonglong(seed)))
        self._seed = seed

    ## @endcond # INCLUDE_INTERNAL


## Quasi-random number generator.
#
# Example:
#
# @code{.py}
#
#   import rocrand
#   import numpy as np
#
#   gen = rocrand.QRNG(rocrand.QRNG.SOBOL32, ndim=4)
#   a = np.empty(1000, np.float32)
#   gen.normal(a, 0.0, 1.0)
#   print(a)
#
# @endcode
class QRNG(RNG):
    ## Default quasi-random generator type, @ref SOBOL32
    DEFAULT           = ROCRAND_RNG_QUASI_DEFAULT
    ## Sobol32 quasi-random generator type
    SOBOL32           = ROCRAND_RNG_QUASI_SOBOL32

    ## @property ndim
    # Mutatable attribute of the number of dimensions of random numbers sequence.
    #
    # Supported values are 1 to 20000.
    # Setting this attribute resets the sequence.

    ## @brief Creates a new quasi-random number generator.
    #
    # A new quasi-random number generator of type @p rngtype is initialized
    # with given @p ndim, @p offset and @p stream.
    #
    # Values of @p rngtype:
    # - @ref DEFAULT
    # - @ref SOBOL32
    #
    # Values if @p ndim are 1 to 20000.
    #
    # @param rngtype Type of quasi-random number generator to create
    # @param ndim    Number of dimensions
    # @param offset  Initial offset of random numbers sequence
    # @param stream  HIP stream for all kernel launches of the generator
    #
    def __init__(self, rngtype=DEFAULT, ndim=None, offset=None, stream=None):
        super(QRNG, self).__init__(rngtype, offset=offset, stream=stream)

        self._ndim = 1
        if ndim is not None:
            self.ndim = ndim

    ## @cond INCLUDE_INTERNAL

    @property
    def ndim(self):
        return self._ndim

    @ndim.setter
    def ndim(self, ndim):
        check_rocrand(rocrand.rocrand_set_quasi_random_generator_dimensions(self._gen, c_uint(ndim)))
        self._ndim = ndim

    ## @endcond # INCLUDE_INTERNAL

## Returns the version number of the rocRAND library.
def get_version():
    version = c_int(0)
    check_rocrand(rocrand.rocrand_get_version(byref(version)))
    return version.value
