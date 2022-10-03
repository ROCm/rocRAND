// Copyright (c) 2017-2022 Advanced Micro Devices, Inc. All rights reserved.
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

#include "benchmark_rocrand_utils.hpp"
#include "cmdparser.hpp"

#include <benchmark/benchmark.h>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include <map>
#include <string>
#include <vector>

#ifndef DEFAULT_RAND_N
const size_t DEFAULT_RAND_N = 1024 * 1024 * 128;
#endif

typedef rocrand_rng_type rng_type_t;
template<typename T>
using generate_func_type = std::function<rocrand_status(rocrand_generator, T*, size_t)>;

template<typename T>
void run_benchmark(benchmark::State&     state,
                   generate_func_type<T> generate_func,
                   const size_t          size,
                   const size_t          trials,
                   const size_t          dimensions,
                   const size_t          offset,
                   const rng_type_t      rng_type,
                   hipStream_t           stream)
{

    const size_t rounded_size = (size / dimensions) * dimensions;

    T* data;
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data), rounded_size * sizeof(T)));

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

    rocrand_status status = rocrand_set_quasi_random_generator_dimensions(generator, dimensions);
    if(status != ROCRAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
    {
        ROCRAND_CHECK(status);
    }

    ROCRAND_CHECK(rocrand_set_stream(generator, stream));

    status = rocrand_set_offset(generator, offset);
    if(status != ROCRAND_STATUS_TYPE_ERROR) // If the RNG is not pseudo-random
    {
        ROCRAND_CHECK(status);
    }

    // Warm-up
    for(size_t i = 0; i < 15; i++)
    {
        ROCRAND_CHECK(generate_func(generator, data, rounded_size));
    }
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    for(auto _ : state)
    {
        HIP_CHECK(hipEventRecord(start, stream));
        for(size_t i = 0; i < trials; i++)
        {
            ROCRAND_CHECK(generate_func(generator, data, rounded_size));
        }
        HIP_CHECK(hipEventRecord(stop, stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float elapsed = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));

        state.SetIterationTime(elapsed / 1000.f);
    }
    state.SetBytesProcessed(trials * state.iterations() * rounded_size * sizeof(T));
    state.SetItemsProcessed(trials * state.iterations() * rounded_size);

    HIP_CHECK(hipEventDestroy(stop));
    HIP_CHECK(hipEventDestroy(start));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    HIP_CHECK(hipFree(data));
}

int main(int argc, char* argv[])
{

    // Parse argv
    benchmark::Initialize(&argc, argv);

    cli::Parser parser(argc, argv);
    parser.set_optional<size_t>("size", "size", DEFAULT_RAND_N, "number of values");
    parser.set_optional<size_t>("dimensions",
                                "dimensions",
                                1,
                                "number of dimensions of quasi-random values");
    parser.set_optional<size_t>("offset", "offset", 0, "offset of generated pseudo-random values");
    parser.set_optional<size_t>("trials", "trials", 20, "number of trials");
    parser.set_optional<std::vector<double>>(
        "lambda",
        "lambda",
        {10.0},
        "space-separated list of lambdas of Poisson distribution");
    parser.run_and_exit_if_error();

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Benchmark info
    add_common_benchmark_rocrand_info();

    const size_t              size            = parser.get<size_t>("size");
    const size_t              trials          = parser.get<size_t>("trials");
    const size_t              dimensions      = parser.get<size_t>("dimensions");
    const size_t              offset          = parser.get<size_t>("offset");
    const std::vector<double> poisson_lambdas = parser.get<std::vector<double>>("lambda");

    benchmark::AddCustomContext("size", std::to_string(size));
    benchmark::AddCustomContext("trials", std::to_string(trials));
    benchmark::AddCustomContext("dimensions", std::to_string(dimensions));
    benchmark::AddCustomContext("offset", std::to_string(offset));

    const std::map<rng_type_t, std::string> engine_type_map{
        {          ROCRAND_RNG_PSEUDO_MTGP32,            "mtgp32"},
#ifndef USE_HIP_CPU
        {         ROCRAND_RNG_PSEUDO_MT19937,           "mt19937"},
#endif
        {          ROCRAND_RNG_PSEUDO_XORWOW,            "xorwow"},
        {        ROCRAND_RNG_PSEUDO_MRG31K3P,          "mrg31k3p"},
        {        ROCRAND_RNG_PSEUDO_MRG32K3A,          "mrg32k3a"},
        {   ROCRAND_RNG_PSEUDO_PHILOX4_32_10,            "philox"},
        {         ROCRAND_RNG_PSEUDO_LFSR113,           "lfsr113"},
        { ROCRAND_RNG_PSEUDO_THREEFRY2_32_20,      "threefry2x32"},
        { ROCRAND_RNG_PSEUDO_THREEFRY2_64_20,      "threefry2x64"},
        { ROCRAND_RNG_PSEUDO_THREEFRY4_32_20,      "threefry4x32"},
        { ROCRAND_RNG_PSEUDO_THREEFRY4_64_20,      "threefry4x64"},
        {          ROCRAND_RNG_QUASI_SOBOL32,           "sobol32"},
        {ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32, "scrambled_sobol32"},
        {          ROCRAND_RNG_QUASI_SOBOL64,           "sobol64"},
        {ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64, "scrambled_sobol64"},

    };

    const std::string benchmark_name_prefix = "device_generate";
    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks = {};
    for(std::pair<rng_type_t, std::string> engine : engine_type_map)
    {

        const rng_type_t  engine_type        = engine.first;
        const std::string name_engine_prefix = benchmark_name_prefix + "<" + engine.second + ",";

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "uniform-uint>").c_str(),
            &run_benchmark<unsigned int>,
            [](rocrand_generator gen, unsigned int* data, size_t size_gen)
            { return rocrand_generate(gen, data, size_gen); },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "uniform-uchar>").c_str(),
            &run_benchmark<unsigned char>,
            [](rocrand_generator gen, unsigned char* data, size_t size_gen)
            { return rocrand_generate_char(gen, data, size_gen); },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "uniform-ushort>").c_str(),
            &run_benchmark<unsigned short>,
            [](rocrand_generator gen, unsigned short* data, size_t size_gen)
            { return rocrand_generate_short(gen, data, size_gen); },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "uniform-half>").c_str(),
            &run_benchmark<__half>,
            [](rocrand_generator gen, __half* data, size_t size_gen)
            { return rocrand_generate_uniform_half(gen, data, size_gen); },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        benchmarks.emplace_back(
            benchmark::RegisterBenchmark((name_engine_prefix + "uniform-float>").c_str(),
                                         &run_benchmark<float>,
                                         [](rocrand_generator gen, float* data, size_t size_gen)
                                         { return rocrand_generate_uniform(gen, data, size_gen); },
                                         size,
                                         trials,
                                         dimensions,
                                         offset,
                                         engine_type,
                                         stream));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "uniform-double>").c_str(),
            &run_benchmark<double>,
            [](rocrand_generator gen, double* data, size_t size_gen)
            { return rocrand_generate_uniform_double(gen, data, size_gen); },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "normal-half>").c_str(),
            &run_benchmark<__half>,
            [](rocrand_generator gen, __half* data, size_t size_gen)
            {
                return rocrand_generate_normal_half(gen,
                                                    data,
                                                    size_gen,
                                                    __float2half(0.0f),
                                                    __float2half(1.0f));
            },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "normal-float>").c_str(),
            &run_benchmark<float>,
            [](rocrand_generator gen, float* data, size_t size_gen)
            { return rocrand_generate_normal(gen, data, size_gen, 0.0f, 1.0f); },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "normal-double>").c_str(),
            &run_benchmark<double>,
            [](rocrand_generator gen, double* data, size_t size_gen)
            { return rocrand_generate_normal_double(gen, data, size_gen, 0.0, 1.0); },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "log-normal-half>").c_str(),
            &run_benchmark<__half>,
            [](rocrand_generator gen, __half* data, size_t size_gen)
            {
                return rocrand_generate_log_normal_half(gen,
                                                        data,
                                                        size_gen,
                                                        __float2half(0.0f),
                                                        __float2half(1.0f));
            },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "log-normal-float>").c_str(),
            &run_benchmark<float>,
            [](rocrand_generator gen, float* data, size_t size_gen)
            { return rocrand_generate_log_normal(gen, data, size_gen, 0.0f, 1.0f); },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (name_engine_prefix + "log-normal-double>").c_str(),
            &run_benchmark<double>,
            [](rocrand_generator gen, double* data, size_t size_gen)
            { return rocrand_generate_log_normal_double(gen, data, size_gen, 0.0, 1.0); },
            size,
            trials,
            dimensions,
            offset,
            engine_type,
            stream));

        for(auto lambda : poisson_lambdas)
        {
            const std::string poisson_dis_name
                = std::string("poisson(lambda=") + std::to_string(lambda) + ")";
            benchmarks.emplace_back(benchmark::RegisterBenchmark(
                (name_engine_prefix + poisson_dis_name).c_str(),
                &run_benchmark<unsigned int>,
                [lambda](rocrand_generator gen, unsigned int* data, size_t size_gen)
                { return rocrand_generate_poisson(gen, data, size_gen, lambda); },
                size,
                trials,
                dimensions,
                offset,
                engine_type,
                stream));
        }
    }
    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }
    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    HIP_CHECK(hipStreamDestroy(stream));

    return 0;
}
