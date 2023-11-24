// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

/// \file
/// \brief This file implements a host/device kernel abstraction.
///
/// rocRAND supports running generators on both the device and the host.
/// Since the generators should generate the same results on both of these,
/// generation kernels are "launched" with the same number of blocks, threads, etc,
/// on the host as they are on the GPU. This file contains abstraction to help with
/// making code generic for both of these "systems".

#ifndef ROCRAND_RNG_SYSTEM_H_
#define ROCRAND_RNG_SYSTEM_H_

#include "common.hpp"
#include "config_types.hpp"
#include "rocrand/rocrand.h"

#include <hip/hip_runtime.h>

#include <new>

#include <stdint.h>

struct rocrand_system_host
{
    static constexpr bool is_device()
    {
        return false;
    }

    template<typename T>
    static rocrand_status alloc(T** ptr, size_t n)
    {
        *ptr = new(std::nothrow) T[n];
        if(!*ptr)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    template<typename T>
    static void free(T* ptr)
    {
        delete[] ptr;
    }

    template<typename... UserArgs>
    struct KernelArgs
    {
        dim3                    num_blocks;
        dim3                    num_threads;
        std::tuple<UserArgs...> user_args;
    };

    template<auto Kernel, size_t... Is, typename... Args>
    static void invoke_kernel(dim3 block,
                              dim3 thread,
                              dim3 grid_dim,
                              dim3 block_dim,
                              std::index_sequence<Is...>,
                              const std::tuple<Args...>& args)
    {
        Kernel(block, thread, grid_dim, block_dim, std::get<Is>(args)...);
    }

    template<auto Kernel,
             typename ConfigProvider = rocrand_host::detail::static_block_size_config_provider<
                 ROCRAND_DEFAULT_MAX_BLOCK_SIZE>,
             typename T     = unsigned int,
             bool IsDynamic = false,
             typename... Args>
    static rocrand_status
        launch(dim3 num_blocks, dim3 num_threads, hipStream_t stream, Args... args)
    {
        (void)IsDynamic; // Not relevant on host launches

        using KernelArgsType = KernelArgs<Args...>;

        const auto kernel_callback = [](void* userdata)
        {
            auto*      kernel_args = reinterpret_cast<KernelArgsType*>(userdata);
            const auto num_blocks  = kernel_args->num_blocks;
            const auto num_threads = kernel_args->num_threads;
            for(uint32_t bz = 0; bz < num_blocks.z; ++bz)
            {
                for(uint32_t by = 0; by < num_blocks.y; ++by)
                {
                    for(uint32_t bx = 0; bx < num_blocks.x; ++bx)
                    {
                        for(uint32_t tz = 0; tz < num_threads.z; ++tz)
                        {
                            for(uint32_t ty = 0; ty < num_threads.y; ++ty)
                            {
                                for(uint32_t tx = 0; tx < num_threads.x; ++tx)
                                {
                                    invoke_kernel<Kernel>(
                                        dim3(bx, by, bz),
                                        dim3(tx, ty, tz),
                                        num_blocks,
                                        num_threads,
                                        std::make_index_sequence<sizeof...(Args)>(),
                                        kernel_args->user_args);
                                }
                            }
                        }
                    }
                }
            }

            delete kernel_args;
        };

        auto* kernel_args
            = new KernelArgsType{num_blocks, num_threads, std::tuple<Args...>(args...)};

        hipError_t status = hipLaunchHostFunc(stream, kernel_callback, kernel_args);

        if(status != hipSuccess)
        {
            // At this point, if the callback has not been invoked, there will be a memory
            // leak. It is unclear whether hipLaunchHostFunc can return an error after the
            // callback has already been invoked, but in such case there would be a double
            // free (crash) instead of a memory leak, so we will just leak it.
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }

        return ROCRAND_STATUS_SUCCESS;
    }
};

namespace detail
{

template<auto Kernel, typename ConfigProvider, typename T, bool IsDynamic, typename... Args>
__global__ __launch_bounds__((rocrand_host::detail::get_block_size<ConfigProvider, T>(
    IsDynamic))) void kernel_wrapper(Args... args)
{
    Kernel(blockIdx, threadIdx, gridDim, blockDim, args...);
}

} // namespace detail

struct rocrand_system_device
{
    static constexpr bool is_device()
    {
        return true;
    }

    template<typename T>
    static rocrand_status alloc(T** ptr, size_t n)
    {
        hipError_t error = hipMalloc(reinterpret_cast<void**>(ptr), sizeof(T) * n);
        if(error != hipSuccess)
        {
            return ROCRAND_STATUS_ALLOCATION_FAILED;
        }
        return ROCRAND_STATUS_SUCCESS;
    }

    template<typename T>
    static void free(T* ptr)
    {
        ROCRAND_HIP_FATAL_ASSERT(hipFree(ptr));
    }

    template<auto Kernel,
             typename ConfigProvider = rocrand_host::detail::static_block_size_config_provider<
                 ROCRAND_DEFAULT_MAX_BLOCK_SIZE>,
             typename T     = unsigned int,
             bool IsDynamic = false,
             typename... Args>
    static rocrand_status
        launch(dim3 num_blocks, dim3 num_threads, hipStream_t stream, Args... args)
    {
        detail::kernel_wrapper<Kernel, ConfigProvider, T, IsDynamic>
            <<<num_blocks, num_threads, 0, stream>>>(args...);
        if(hipGetLastError() != hipSuccess)
        {
            return ROCRAND_STATUS_LAUNCH_FAILURE;
        }
        return ROCRAND_STATUS_SUCCESS;
    }
};

#endif
