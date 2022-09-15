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

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <utility>
#include <algorithm>

// Google benchmark
#include <benchmark/benchmark.h>

#include "cmdparser.hpp"

#include <cuda_runtime.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#ifndef DEFAULT_RAND_N
const size_t DEFAULT_RAND_N = 1024 * 1024 * 128;
#endif

typedef curandRngType rng_type_t;

template<typename T>
using generate_func_type = std::function<curandStatus_t(curandGenerator_t, T *, size_t)>;

template<typename T>
void run_benchmark(benchmark::State&     state,
                   const rng_type_t      rng_type,
                   generate_func_type<T> generate_func,
                   const size_t          size,
                   const size_t          trials,
                   const size_t          offset,
                   const size_t          dimensions)
{
    T * data;
    CUDA_CALL(cudaMalloc((void **)&data, size * sizeof(T)));

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, rng_type));

    curandStatus_t status = curandSetQuasiRandomGeneratorDimensions(generator, dimensions);
    if (status != CURAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
    {
        CURAND_CALL(status);
    }

    status = curandSetGeneratorOffset(generator, offset);
    if (status != CURAND_STATUS_TYPE_ERROR) // If the RNG is not pseudo-random
    {
        CURAND_CALL(status);
    }

    // Warm-up
    for (size_t i = 0; i < 5; i++)
    {
        CURAND_CALL(generate_func(generator, data, size));
    }
    CUDA_CALL(cudaDeviceSynchronize());

    // Measurement
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < trials; i++)
    {
        CURAND_CALL(generate_func(generator, data, size));
    }
    CUDA_CALL(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    CURAND_CALL(curandDestroyGenerator(generator));
    CUDA_CALL(cudaFree(data));
}

void configure_parser(cli::Parser& parser)
{
    parser.set_optional<size_t>("size", "size", DEFAULT_RAND_N, "number of values");
    parser.set_optional<size_t>("trials", "trials", 20, "number of trials");
    parser.set_optional<size_t>("offset", "offset", 0, "offset of generated pseudo-random values");
    parser.set_optional<size_t>("dimensions",
                                "dimensions",
                                1,
                                "number of dimensions of quasi-random values");
    parser.set_optional<std::vector<double>>(
        "lambda",
        "lambda",
        {10.0},
        "space-separated list of lambdas of Poisson distribution");
}

int main(int argc, char* argv[])
{
    benchmark::Initialize(&argc, argv)

    // Parse arguments from command line
    cli::Parser parser(argc, argv);
    configure_parser(parser);
    parser.run_and_exit_if_error();

    add_common_benchmark_info();

    const size_t              size            = parser.get<size_t>("size");
    const size_t              trials          = parser.get<size_t>("trials");
    const size_t              offset          = parser.get<size_t>("offset");
    const size_t              dimensions      = parser.get<size_t>("dimensions");
    const std::vector<double> poisson_lambdas = parser.get<std::vector<double>>("lambda");

    benchmark::AddCustomContext("size", std::to_string(size));
    benchmark::AddCustomContext("trials", std::to_string(trials));
    benchmark::AddCustomContext("offset", std::to_string(offset));
    benchmark::AddCustomContext("dimensions", std::to_string(dimensions));

    const std::map<rng_type_t, std::string> engine_type_map{
        {          CURAND_RNG_PSEUDO_MTGP32,            "mtgp32"},
        {        CURAND_RNG_PSEUDO_MRG32K3A,          "mrg32k3a"},
        {         CURAND_RNG_PSEUDO_MT19937,           "mt19937"},
        {   CURAND_RNG_PSEUDO_PHILOX4_32_10,            "philox"},
        {CURAND_RNG_QUASI_SCRAMBLED_SOBOL32, "scrambled_sobol32"},
        {CURAND_RNG_QUASI_SCRAMBLED_SOBOL64, "scrambled_sobol64"},
        {          CURAND_RNG_QUASI_SOBOL32,           "sobol32"},
        {          CURAND_RNG_QUASI_SOBOL64,           "sobol64"},
        {          CURAND_RNG_PSEUDO_XORWOW,            "xorwow"},
    };

    const std::string                            benchmark_name_prefix = "device_generate";
    std::vector<benchmark::internal::Benchmark*> benchmarks            = {};

    // Add benchmarks
    for(std::pair<rng_type_t, std::string> engine : engine_type_map)
    {
        const rng_type_t  engine_type           = engine.first;
        const std::string benchmark_name_engine = benchmark_name_prefix + "<" + engine.second + ",";

        if(engine_type != CURAND_RNG_QUASI_SOBOL64
           && engine_type != CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
            benchmarks.emplace_back(benchmark::RegisterBenchmark(
                (benchmark_name_engine + "uniform-uint>").c_str(),
                &run_benchmark<unsigned int>,
                engine_type,
                [](curandGenerator_t gen, unsigned int* data, size_t size)
                { return curandGenerate(gen, data, size); },
                size,
                trials,
                offset,
                dimensions));
        else
            benchmarks.emplace_back(benchmark::RegisterBenchmark(
                (benchmark_name_engine + "uniform-long-long>").c_str(),
                &run_benchmark<unsigned long long>,
                engine_type,
                [](curandGenerator_t gen, unsigned long long* data, size_t size)
                { return curandGenerateLongLong(gen, data, size); },
                size,
                trials,
                offset,
                dimensions));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
                (benchmark_name_engine + "uniform-float>").c_str(),
                &run_benchmark<float>,
                engine_type,
                [](curandGenerator_t gen, float* data, size_t size)
                { return curandGenerateUniform(gen, data, size); },
                size,
                trials,
                offset,
                dimensions));
        
        benchmarks.emplace_back(benchmark::RegisterBenchmark(
                (benchmark_name_engine + "uniform-double>").c_str(),
                &run_benchmark<double>,
                engine_type,
                [](curandGenerator_t gen, double* data, size_t size)
                { return curandGenerateUniformDouble(gen, data, size); },
                size,
                trials,
                offset,
                dimensions));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (benchmark_name_engine + "normal-float>").c_str(),
            &run_benchmark<float>,
            engine_type,
            [](curandGenerator_t gen, float* data, size_t size)
            { return curandGenerateNormal(gen, data, size, 0.0f, 1.0f); },
            size,
            trials,
            offset,
            dimensions));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (benchmark_name_engine + "normal-double>").c_str(),
            &run_benchmark<double>,
            engine_type,
            [](curandGenerator_t gen, double* data, size_t size)
            { return curandGenerateNormalDouble(gen, data, size, 0.0, 1.0); },
            size,
            trials,
            offset,
            dimensions));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (benchmark_name_engine + "log-normal-float>").c_str(),
            &run_benchmark<float>,
            engine_type,
            [](curandGenerator_t gen, float* data, size_t size)
            { return curandGenerateLogNormal(gen, data, size, 0.0f, 1.0f); },
            size,
            trials,
            offset,
            dimensions));

        benchmarks.emplace_back(benchmark::RegisterBenchmark(
            (benchmark_name_engine + "log-normal-double>").c_str(),
            &run_benchmark<double>,
            engine_type,
            [](curandGenerator_t gen, double* data, size_t size)
            { return curandGenerateLogNormalDouble(gen, data, size, 0.0, 1.0); },
            size,
            trials,
            offset,
            dimensions));

        for(auto lambda : poisson_lambdas)
        {
            const std::string poisson_dis_name
                = std::string("poisson(lambda=") + std::to_string(lambda) + ")>";

            benchmarks.emplace_back(benchmark::RegisterBenchmark(
                (benchmark_name_engine + poisson_dis_name).c_str(),
                &run_benchmark<unsigned int>,
                engine_type,
                [lambda](curandGenerator_t gen, unsigned int* data, size_t size)
                { return curandGeneratePoisson(gen, data, size, lambda); },
                size,
                trials,
                offset,
                dimensions));
        }
    }
    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    benchmark::RunSpecifiedBenchmarks();

    return 0;
}
