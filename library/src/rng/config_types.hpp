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

#ifndef ROCRAND_RNG_CONFIG_TYPES_H_
#define ROCRAND_RNG_CONFIG_TYPES_H_

#include "common.hpp"
#include "rocrand/rocrand.h"

#include <hip/hip_runtime.h>

#include <atomic>
#include <cassert>
#include <limits>
#include <string>

namespace rocrand_host
{
namespace detail
{

/// \brief Represents a device target's processor architecture.
enum class target_arch : unsigned int
{
    // This must be zero, to initialize the device -> architecture cache
    invalid = 0,
    gfx900  = 900,
    gfx902  = 902,
    gfx904  = 904,
    gfx906  = 906,
    gfx908  = 908,
    gfx909  = 909,
    gfx90a  = 910,
    gfx1030 = 1030,
    gfx1100 = 1100,
    gfx1101 = 1101,
    gfx1102 = 1102,
    unknown = std::numeric_limits<unsigned int>::max(),
};

/// @brief Returns the detected processor architecture of the device that is currently compiled against.
/// Note, that this will always return \ref target_arch::unknown during host compilation.
/// On the host, use the other overloads.
__host__ __device__ constexpr target_arch get_device_arch()
{
#if defined(__gfx900__)
    return target_arch::gfx900;
#elif defined(__gfx902__)
    return target_arch::gfx902;
#elif defined(__gfx904__)
    return target_arch::gfx904;
#elif defined(__gfx906__)
    return target_arch::gfx906;
#elif defined(__gfx908__)
    return target_arch::gfx908;
#elif defined(__gfx909__)
    return target_arch::gfx909;
#elif defined(__gfx90a__)
    return target_arch::gfx90a;
#elif defined(__gfx1030__)
    return target_arch::gfx1030;
#elif defined(__gfx1100__)
    return target_arch::gfx1100;
#elif defined(__gfx1101__)
    return target_arch::gfx1101;
#elif defined(__gfx1102__)
    return target_arch::gfx1102;
#else
    return target_arch::unknown;
#endif
}

/// @brief Maps the device architecture name to the \ref target_arch enum.
/// @param arch_name The name of the architecture. Normally queried from the HIP runtime.
/// @return The detected element of the enum, of \ref target_arch::unknown, if not found.
inline target_arch parse_gcn_arch(const std::string& arch_name)
{
    const std::string target_names[]         = {"gfx900",
                                                "gfx902",
                                                "gfx904",
                                                "gfx906",
                                                "gfx908",
                                                "gfx909",
                                                "gfx90a",
                                                "gfx1030",
                                                "gfx1100",
                                                "gfx1101",
                                                "gfx1102"};
    const target_arch target_architectures[] = {
        target_arch::gfx900,
        target_arch::gfx902,
        target_arch::gfx904,
        target_arch::gfx906,
        target_arch::gfx908,
        target_arch::gfx909,
        target_arch::gfx90a,
        target_arch::gfx1030,
        target_arch::gfx1100,
        target_arch::gfx1101,
        target_arch::gfx1102,
    };
    static_assert(sizeof(target_names) / sizeof(target_names[0])
                      == sizeof(target_architectures) / sizeof(target_architectures[0]),
                  "target_names and target_architectures should have the same number of elements");
    constexpr auto num_architectures = sizeof(target_names) / sizeof(target_names[0]);

    for(unsigned int i = 0; i < num_architectures; ++i)
    {
        if(arch_name.find(target_names[i]) == 0)
        {
            return target_architectures[i];
        }
    }
    return target_arch::unknown;
}

/// @brief Queries the device architecture of a HIP device.
/// @param device_id The ID of the HIP device.
/// @param arch Out parameter. The detected device architecture is set here.
/// @return \ref hipSuccess if the querying was successful, a different error code otherwise.
inline hipError_t get_device_arch(int device_id, target_arch& arch)
{
#ifdef USE_DEVICE_DISPATCH
    constexpr unsigned int          device_arch_cache_size             = 512;
    static std::atomic<target_arch> arch_cache[device_arch_cache_size] = {};

    assert(device_id >= 0);
    if(static_cast<unsigned int>(device_id) >= device_arch_cache_size)
    {
        // Device architecture cache is too small.
        return hipErrorUnknown;
    }

    arch = arch_cache[device_id].load(std::memory_order_relaxed);
    if(arch != target_arch::invalid)
    {
        return hipSuccess;
    }

    hipDeviceProp_t  device_props;
    const hipError_t result = hipGetDeviceProperties(&device_props, device_id);
    if(result != hipSuccess)
    {
        return result;
    }

    arch = parse_gcn_arch(device_props.gcnArchName);
    arch_cache[device_id].exchange(arch, std::memory_order_relaxed);
#else
    arch = target_arch::unknown;
#endif

    return hipSuccess;
}

#ifdef USE_DEVICE_DISPATCH
/// @brief Queries a device corresponding to a stream using the HIP API.
/// @param stream The stream in question.
/// @param device_id Out parameter. The result of the device query is written here.
/// @return \ref hipSuccess if the querying was successful, a different error code otherwise.
inline hipError_t get_device_from_stream(const hipStream_t stream, int& device_id)
{
    constexpr hipStream_t default_stream = 0;
    if(stream == default_stream || stream == hipStreamPerThread)
    {
        const hipError_t result = hipGetDevice(&device_id);
        if(result != hipSuccess)
        {
            return result;
        }
        return hipSuccess;
    }
    device_id = hipGetStreamDeviceId(stream);
    if(device_id < 0)
    {
        return hipErrorInvalidHandle;
    }
    return hipSuccess;
}
#endif

/// @brief Queries the device architecture of a device corresponding to a HIP stream.
/// @param stream The ID of the HIP stream.
/// @param arch Out parameter. The detected device architecture is set here.
/// @return \ref hipSuccess if the querying was successful, a different error code otherwise.
inline hipError_t get_device_arch(const hipStream_t stream, target_arch& arch)
{
#ifdef USE_DEVICE_DISPATCH
    int              device_id;
    const hipError_t result = get_device_from_stream(stream, device_id);
    if(result != hipSuccess)
    {
        return result;
    }
    return get_device_arch(device_id, arch);
#else
    (void)stream;
    arch = target_arch::unknown;
    return hipSuccess;
#endif
}

/// @brief Default kernel launch configuration for the random engines.
/// Partial or full template specializations are implemented for the engines
/// to select different parameterizations.
/// Architecture-dependent (dynamic) configurations can be implemented by
/// specializing \ref generator_config_selector.
/// @tparam T The type of the generated random values.
/// @tparam GeneratorType The kind of the random engine.
template<rocrand_rng_type GeneratorType, class T>
struct generator_config_defaults
{
    // The predicate is always false, but it must depend on the type parameters.
    static_assert(sizeof(T) == 0,
                  "generator_config_defaults must be specialized for each generator type.");
};

/// @brief Provides architecture-dependent (dynamic) kernel launch
/// configuration to the random engines.
/// Partial or full template specializations are implemented for the engines
/// to select different parameterizations.
/// @tparam T The type of the generated random values.
/// @tparam GeneratorType The kind of the random engine.
template<rocrand_rng_type GeneratorType, class T>
struct generator_config_selector
{
    __host__ __device__ constexpr unsigned int get_threads(const target_arch /*arch*/) const
    {
        return generator_config_defaults<GeneratorType, T>{}.threads;
    }

    __host__ __device__ constexpr unsigned int get_blocks(const target_arch /*arch*/) const
    {
        return generator_config_defaults<GeneratorType, T>{}.blocks;
    }
};

/// @brief POD struct to store kernel launch configurations for the random engines.
/// This struct can be used both at runtime on the host and at compile time on the device.
struct generator_config
{
    unsigned int threads;
    unsigned int blocks;
};

/// @brief Returns whether the provided ordering allows the architecture-dependent
/// selection of kernel launch parameters.
inline bool is_ordering_dynamic(const rocrand_ordering ordering)
{
    return ordering == ROCRAND_ORDERING_PSEUDO_DYNAMIC
           || ordering == ROCRAND_ORDERING_QUASI_DEFAULT;
}

/// @brief Selects the preset kernel launch config for the given random engine and
/// generated value type.
/// @tparam T The type of the generated random values.
/// @tparam GeneratorType The kind of the random engine.
/// @param stream The HIP stream on which the random generation is executed.
/// @param ordering The ordering of the random engine.
/// @param config Out parameter. The selected launch config is written here.
/// @return \ref hipSuccess if the querying was successful, a different error code otherwise.
template<rocrand_rng_type GeneratorType, class T>
hipError_t get_generator_config(const hipStream_t      stream,
                                const rocrand_ordering ordering,
                                generator_config&      config)
{
    if(is_ordering_dynamic(ordering))
    {
        target_arch      current_arch;
        const hipError_t error = get_device_arch(stream, current_arch);
        if(error != hipSuccess)
        {
            return error;
        }
        config.threads = generator_config_selector<GeneratorType, T>{}.get_threads(current_arch);
        config.blocks  = generator_config_selector<GeneratorType, T>{}.get_blocks(current_arch);
    }
    else
    {
        config.threads = generator_config_defaults<GeneratorType, T>{}.threads;
        config.blocks  = generator_config_defaults<GeneratorType, T>{}.blocks;
    }
    return hipSuccess;
}

/// @brief Selects the preset kernel launch config for the given random engine and
/// generated value type.
/// @tparam T The type of the generated random values.
/// @tparam GeneratorType The kind of the random engine.
/// @param dynamic_config Whether architecture-specific launch config can be selected or not.
/// @return The selected launch config.
template<rocrand_rng_type GeneratorType, class T>
__device__ constexpr generator_config get_generator_config_device(bool dynamic_config)
{
    return generator_config{generator_config_selector<GeneratorType, T>{}.get_threads(
                                dynamic_config ? get_device_arch() : target_arch::unknown),
                            generator_config_selector<GeneratorType, T>{}.get_blocks(
                                dynamic_config ? get_device_arch() : target_arch::unknown)};
}

} // end namespace detail
} // end namespace rocrand_host

#endif // ROCRAND_RNG_CONFIG_TYPES_H_