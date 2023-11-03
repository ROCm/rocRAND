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

template<class ConfigProvider>
class rocrand_lfsr113_template;

template<class ConfigProvider>
class rocrand_mrg31k3p_template;

template<class ConfigProvider>
class rocrand_mrg32k3a_template;

template<class ConfigProvider>
class rocrand_mtgp32_template;

template<class ConfigProvider>
class rocrand_philox4x32_10_template;

template<class ConfigProvider>
class rocrand_threefry2x32_20_template;

template<class ConfigProvider>
class rocrand_threefry2x64_20_template;

template<class ConfigProvider>
class rocrand_threefry4x32_20_template;

template<class ConfigProvider>
class rocrand_threefry4x64_20_template;

template<class ConfigProvider>
class rocrand_xorwow_template;

namespace benchmark_tuning
{

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

extern template void add_all_benchmarks_for_generator<rocrand_lfsr113_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_mrg31k3p_template>(
    std::vector<benchmark::internal::Benchmark*>& benchmarks, const benchmark_config& config);

extern template void add_all_benchmarks_for_generator<rocrand_mrg32k3a_template>(
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
