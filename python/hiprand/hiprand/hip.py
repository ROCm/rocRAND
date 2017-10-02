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

## @addtogroup hiprandpython
# @{

## @namespace hiprand.hip
# Minimal HIP wrapper

## @}

import os
import ctypes
import ctypes.util
from ctypes import *

import numbers
import numpy as np

from .utils import find_library


## Run-time HIP error.
class HipError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)

## @cond INCLUDE_INTERNAL

hipSuccess = 0
hipMemcpyDeviceToHost = 2

def check_hip(status):
    if status != hipSuccess:
        raise HipError(status)

hip = None

HIP_PATHS = [
    os.getenv("ROCM_PATH"),
    os.getenv("HIP_PATH"),
    "/opt/rocm",
    "/opt/rocm/hip"]

CUDA_PATHS = [
    os.getenv("CUDA_PATH"),
    "/opt/cuda"]

def load_hip():
    global hip

    loading_errors = []
    if hip is None:
        try:
            hip = CDLL(find_library(HIP_PATHS, "libhip_hcc.so"), mode=RTLD_GLOBAL)
        except OSError as e:
            loading_errors.append(str(e))

    if hip is None:
        try:
            cuda = CDLL(find_library(CUDA_PATHS, "libcudart.so"), mode=RTLD_GLOBAL)
        except OSError as e:
            loading_errors.append(str(e))
        else:
            hip = cuda
            # Aliases
            hip.hipMalloc = cuda.cudaMalloc
            hip.hipFree = cuda.cudaFree
            hip.hipMemcpy = cuda.cudaMemcpy

    if hip is None:
        raise ImportError("both libcudart.so and libhip_hcc.so cannot be loaded: " +
                ", ".join(loading_errors))

def hip_malloc(nbytes):
    ptr = c_void_p()
    check_hip(hip.hipMalloc(byref(ptr), c_size_t(nbytes)))
    return ptr

def hip_free(ptr):
    check_hip(hip.hipFree(ptr))

def hip_copy_to_host(dst, src, nbytes):
    check_hip(hip.hipMemcpy(dst, src, c_size_t(nbytes), hipMemcpyDeviceToHost))

class MemoryPointer(object):
    def __init__(self, nbytes):
        self.ptr = None
        self.ptr = hip_malloc(nbytes)

    def __del__(self):
        if self.ptr:
            hip_free(self.ptr)

def device_pointer(dary):
    return dary.data.ptr

## @endcond # INCLUDE_INTERNAL

## Device-side array
class DeviceNDArray(object):
    def __init__(self, shape, dtype, data=None):
        dtype = np.dtype(dtype)

        if isinstance(shape, numbers.Integral):
            shape = (shape,)
        shape = tuple(shape)

        size = np.prod(shape)

        self.shape = shape
        self.dtype = dtype
        self.size = size
        self.nbytes = self.dtype.itemsize * self.size

        if data is None:
            self.data = MemoryPointer(self.nbytes)
        else:
            self.data = data

    def copy_to_host(self, ary=None):
        if ary is None:
            ary = np.empty(self.shape, self.dtype)
        else:
            if self.dtype != ary.dtype:
                raise TypeError("self and ary must have the same dtype")
            if self.size > ary.size:
                raise ValueError("size of self must be less than size of ary")

        dst = ary.ctypes.data_as(c_void_p)
        src = device_pointer(self)
        hip_copy_to_host(dst, src, self.nbytes)

        return ary

## Create an empty device-side array
#
# @param shape Shape of the array (see @c numpy.ndarray.shape)
# @param dtype Type of the array (see @c numpy.ndarray.dtype)
def empty(shape, dtype):
    return DeviceNDArray(shape, dtype)
