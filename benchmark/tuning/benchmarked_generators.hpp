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

/// \file benchmarked_generators.hpp
/// This header should contain the benchmarking setup for specific rocRAND generators.
/// E.g. specialization of the \c output_type_supported class template.
/// Additionally, an \c extern \c template declaration could be present for each
/// benchmarked generator, to offload compilation of the benchmarks to multiple translation
/// units.

#ifndef ROCRAND_BENCHMARK_TUNING_BENCHMARKED_GENERATORS_HPP_
#define ROCRAND_BENCHMARK_TUNING_BENCHMARKED_GENERATORS_HPP_

#include "benchmark_tuning.hpp"
#include "rng/system.hpp"

#include "rocrand/rocrand_mt19937_precomputed.h"

// Forward declaring generator templates, so includes can be omitted.
// The tuning benchmarks are instantiated in the respective benchmark_tuning_*.cpp
// source files.

// mt19937 needs to be included, as access to threads_per_generator is needed
#include "rng/mt19937.hpp"

template<class ConfigProvider>
class rocrand_lfsr113_template;

template<class System, class ConfigProvider>
class rocrand_mrg31k3p_template;

template<class ConfigProvider>
class rocrand_mrg32k3a_template;

template<class ConfigProvider>
class rocrand_mtgp32_template;

template<class System, class ConfigProvider>
class rocrand_philox4x32_10_template;

template<class Engine, class ConfigProvider>
class rocrand_threefry_template;

template<class System, class ConfigProvider>
class rocrand_xorwow_template;

// Further forward declarations
namespace rocrand_device
{
class threefry2x32_20_engine;
class threefry2x64_20_engine;
class threefry4x32_20_engine;
class threefry4x64_20_engine;
} // namespace rocrand_device

namespace rocrand_host::detail
{
template<class DeviceEngine>
struct threefry_device_engine;
} // namespace rocrand_host::detail

namespace benchmark_tuning
{

// Defining aliases for all generator templates in the benchmark_tuning namespace,
// so the device system can be selected, for the generators that already implement
// both host and device systems

template<class ConfigProvider>
using rocrand_lfsr113_template = ::rocrand_lfsr113_template<ConfigProvider>;

template<class ConfigProvider>
using rocrand_mrg31k3p_template
    = ::rocrand_mrg31k3p_template<rocrand_system_device, ConfigProvider>;

template<class ConfigProvider>
using rocrand_mrg32k3a_template = ::rocrand_mrg32k3a_template<ConfigProvider>;

template<class ConfigProvider>
using rocrand_mtgp32_template = ::rocrand_mtgp32_template<ConfigProvider>;

template<class ConfigProvider>
using rocrand_mt19937_template = ::rocrand_mt19937_template<ConfigProvider>;

template<class ConfigProvider>
using rocrand_philox4x32_10_template
    = ::rocrand_philox4x32_10_template<rocrand_system_device, ConfigProvider>;

template<class ConfigProvider>
using rocrand_threefry2x32_20_template = ::rocrand_threefry_template<
    rocrand_host::detail::threefry_device_engine<rocrand_device::threefry2x32_20_engine>,
    ConfigProvider>;

template<class ConfigProvider>
using rocrand_threefry2x64_20_template = ::rocrand_threefry_template<
    rocrand_host::detail::threefry_device_engine<rocrand_device::threefry2x64_20_engine>,
    ConfigProvider>;

template<class ConfigProvider>
using rocrand_threefry4x32_20_template = ::rocrand_threefry_template<
    rocrand_host::detail::threefry_device_engine<rocrand_device::threefry4x32_20_engine>,
    ConfigProvider>;

template<class ConfigProvider>
using rocrand_threefry4x64_20_template = ::rocrand_threefry_template<
    rocrand_host::detail::threefry_device_engine<rocrand_device::threefry4x64_20_engine>,
    ConfigProvider>;

template<class ConfigProvider>
using rocrand_xorwow_template = ::rocrand_xorwow_template<rocrand_system_device, ConfigProvider>;

template<>
struct output_type_supported<unsigned long long, rocrand_lfsr113_template> : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, rocrand_mrg31k3p_template> : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, rocrand_mrg32k3a_template> : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, rocrand_mtgp32_template> : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, rocrand_mt19937_template> : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, rocrand_philox4x32_10_template>
    : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, rocrand_threefry2x32_20_template>
    : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, rocrand_threefry4x32_20_template>
    : public std::false_type
{};

template<>
struct output_type_supported<unsigned long long, rocrand_xorwow_template> : public std::false_type
{};

template<class T>
struct config_filter<rocrand_mtgp32_template, T>
{
    static constexpr bool is_enabled(rocrand_host::detail::generator_config config)
    {
        return config.blocks <= 512;
    }
};

template<class T>
struct config_filter<rocrand_mt19937_template, T>
{
    static constexpr bool is_enabled(rocrand_host::detail::generator_config config)
    {
        return (config.blocks * config.threads / rocrand_mt19937::threads_per_generator)
               <= mt19937_jumps_radix * mt19937_jumps_radix;
    }
};

template<>
struct distribution_input<rocrand_threefry2x64_20_template>
{
    using type = unsigned long long;
};

template<>
struct distribution_input<rocrand_threefry4x64_20_template>
{
    using type = unsigned long long;
};

extern template void add_all_benchmarks_for_generator<rocrand_lfsr113_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_mrg31k3p_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_mrg32k3a_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_mt19937_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_mtgp32_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_philox4x32_10_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_threefry2x32_20_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_threefry2x64_20_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_threefry4x32_20_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_threefry4x64_20_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_xorwow_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

} // namespace benchmark_tuning

#endif
