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

#include "rng/common.hpp"
#include "rng/config_types.hpp"
#include "test_common.hpp"
#include <gtest/gtest.h>

__global__ void write_target_arch(rocrand_host::detail::target_arch* dest_arch)
{
    static constexpr auto arch = rocrand_host::detail::get_device_arch();
    *dest_arch                 = arch;
}

static constexpr rocrand_rng_type dummy_rng_type = rocrand_rng_type(0);

namespace rocrand_host
{
namespace detail
{

template<class T>
struct generator_config_defaults<dummy_rng_type, T>
{
    static constexpr unsigned int threads = 256;
    static constexpr unsigned int blocks  = 1;
};

} // end namespace detail
} // end namespace rocrand_host

template<class T,
         unsigned int BlockSize
         = rocrand_host::detail::get_generator_config_device<dummy_rng_type, T>(true).threads>
__global__ __launch_bounds__(BlockSize) void write_config(unsigned int* block_size,
                                                          unsigned int* grid_size)
{
    if(blockIdx.x == 0 && threadIdx.x == 0 && BlockSize == blockDim.x)
    {
        *block_size = blockDim.x;
        *grid_size  = gridDim.x;
    }
}

TEST(rocrand_config_dispatch_tests, host_matches_device)
{
    const hipStream_t stream = 0;

    rocrand_host::detail::target_arch host_arch;
    HIP_CHECK(rocrand_host::detail::get_device_arch(stream, host_arch));

    rocrand_host::detail::target_arch* device_arch_ptr;
    HIP_CHECK(hipMallocHelper(&device_arch_ptr, sizeof(*device_arch_ptr)));

    hipLaunchKernelGGL(write_target_arch, dim3(1), dim3(1), 0, stream, device_arch_ptr);
    HIP_CHECK(hipGetLastError());

    rocrand_host::detail::target_arch device_arch;
    HIP_CHECK(hipMemcpy(&device_arch, device_arch_ptr, sizeof(device_arch), hipMemcpyDeviceToHost));

    ASSERT_NE(host_arch, rocrand_host::detail::target_arch::invalid);
    ASSERT_EQ(host_arch, device_arch);
}

TEST(rocrand_config_dispatch_tests, parse_common_architectures)
{
    using rocrand_host::detail::parse_gcn_arch;
    using rocrand_host::detail::target_arch;

    ASSERT_EQ(parse_gcn_arch(""), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch("not a gfx arch"), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch(":"), target_arch::unknown);
    ASSERT_EQ(parse_gcn_arch("g:"), target_arch::unknown);

    ASSERT_EQ(parse_gcn_arch("gfx900:sramecc+"), target_arch::gfx900);
    ASSERT_EQ(parse_gcn_arch("gfx906:::"), target_arch::gfx906);
    ASSERT_EQ(parse_gcn_arch("gfx908:"), target_arch::gfx908);
    ASSERT_EQ(parse_gcn_arch("gfx90a:sramecc+:xnack-"), target_arch::gfx90a);
}

TEST(rocrand_config_dispatch_tests, get_config_on_host_and_device)
{
    using T = unsigned int;

    const hipStream_t stream = 0;

    unsigned int *d_block_size{}, *d_grid_size{};
    HIP_CHECK(hipMallocHelper(&d_block_size, sizeof(*d_block_size)));
    HIP_CHECK(hipMallocHelper(&d_grid_size, sizeof(*d_grid_size)));

    rocrand_host::detail::generator_config config{};
    const hipError_t error = rocrand_host::detail::get_generator_config<dummy_rng_type, T>(
        stream,
        ROCRAND_ORDERING_PSEUDO_DEFAULT,
        config);
    HIP_CHECK(error);

    hipLaunchKernelGGL(write_config<T>,
                       dim3(config.blocks),
                       dim3(config.threads),
                       0,
                       stream,
                       d_block_size,
                       d_grid_size);
    HIP_CHECK(hipGetLastError());

    unsigned int block_size{}, grid_size{};
    HIP_CHECK(hipMemcpy(&block_size, d_block_size, sizeof(block_size), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&grid_size, d_grid_size, sizeof(grid_size), hipMemcpyDeviceToHost));

    hipFree(d_block_size);
    hipFree(d_grid_size);

    ASSERT_EQ(block_size, config.threads);
    ASSERT_EQ(grid_size, config.blocks);
}

#ifdef USE_DEVICE_DISPATCH
TEST(rocrand_config_dispatch_tests, device_id_from_stream)
{
    using rocrand_host::detail::get_device_from_stream;
    hipDevice_t device_id;
    HIP_CHECK(hipGetDevice(&device_id));

    int                          result;
    static constexpr hipStream_t default_stream = 0;
    HIP_CHECK(get_device_from_stream(default_stream, result));
    ASSERT_EQ(result, device_id);

    HIP_CHECK(get_device_from_stream(hipStreamPerThread, result));
    ASSERT_EQ(result, device_id);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    HIP_CHECK(get_device_from_stream(stream, result));
    HIP_CHECK(hipStreamDestroy(stream));
    ASSERT_EQ(result, device_id);

    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    HIP_CHECK(get_device_from_stream(stream, result));
    HIP_CHECK(hipStreamDestroy(stream));
    ASSERT_EQ(result, device_id);
}
#endif
