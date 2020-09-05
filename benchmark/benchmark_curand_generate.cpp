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

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "cmdparser.hpp"

#include <cuda_runtime.h>
#include <curand.h>

#define CUDA_CALL(x)                                        \
    do                                                      \
    {                                                       \
        if((x) != cudaSuccess)                              \
        {                                                   \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while(0)
#define CURAND_CALL(x)                                      \
    do                                                      \
    {                                                       \
        if((x) != CURAND_STATUS_SUCCESS)                    \
        {                                                   \
            printf("Error at %s:%d\n", __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while(0)

#ifndef DEFAULT_RAND_N
const size_t DEFAULT_RAND_N = 1024 * 1024 * 128;
#endif

typedef curandRngType rng_type_t;

template <typename T>
using generate_func_type = std::function<curandStatus_t(curandGenerator_t, T*, size_t)>;

template <typename T>
void run_benchmark(const cli::Parser&    parser,
                   const rng_type_t      rng_type,
                   generate_func_type<T> generate_func)
{
    const size_t size   = parser.get<size_t>("size");
    const size_t trials = parser.get<size_t>("trials");

    T* data;
    CUDA_CALL(cudaMalloc((void**)&data, size * sizeof(T)));

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, rng_type));

    const size_t   dimensions = parser.get<size_t>("dimensions");
    curandStatus_t status     = curandSetQuasiRandomGeneratorDimensions(generator, dimensions);
    if(status != CURAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
    {
        CURAND_CALL(status);
    }

    // Warm-up
    for(size_t i = 0; i < 5; i++)
    {
        CURAND_CALL(generate_func(generator, data, size));
    }
    CUDA_CALL(cudaDeviceSynchronize());

    // Measurement
    auto start = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < trials; i++)
    {
        CURAND_CALL(generate_func(generator, data, size));
    }
    CUDA_CALL(cudaDeviceSynchronize());
    auto                                      end     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << std::fixed << std::setprecision(3) << "      "
              << "Throughput = " << std::setw(8)
              << (trials * size * sizeof(T)) / (elapsed.count() / 1e3 * (1 << 30))
              << " GB/s, Samples = " << std::setw(8)
              << (trials * size) / (elapsed.count() / 1e3 * (1 << 30))
              << " GSample/s, AvgTime (1 trial) = " << std::setw(8) << elapsed.count() / trials
              << " ms, Time (all) = " << std::setw(8) << elapsed.count() << " ms, Size = " << size
              << std::endl;

    CURAND_CALL(curandDestroyGenerator(generator));
    CUDA_CALL(cudaFree(data));
}

void run_benchmarks(const cli::Parser& parser,
                    const rng_type_t   rng_type,
                    const std::string& distribution)
{
    if(distribution == "uniform-uint")
    {
        if(rng_type != CURAND_RNG_QUASI_SOBOL64 && rng_type != CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
        {
            run_benchmark<unsigned int>(
                parser, rng_type, [](curandGenerator_t gen, unsigned int* data, size_t size) {
                    return curandGenerate(gen, data, size);
                });
        }
    }
    if(distribution == "uniform-long-long")
    {
        if(rng_type == CURAND_RNG_QUASI_SOBOL64 || rng_type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
        {
            run_benchmark<unsigned long long>(
                parser, rng_type, [](curandGenerator_t gen, unsigned long long* data, size_t size) {
                    return curandGenerateLongLong(gen, data, size);
                });
        }
    }
    if(distribution == "uniform-float")
    {
        run_benchmark<float>(parser, rng_type, [](curandGenerator_t gen, float* data, size_t size) {
            return curandGenerateUniform(gen, data, size);
        });
    }
    if(distribution == "uniform-double")
    {
        run_benchmark<double>(
            parser, rng_type, [](curandGenerator_t gen, double* data, size_t size) {
                return curandGenerateUniformDouble(gen, data, size);
            });
    }
    if(distribution == "normal-float")
    {
        run_benchmark<float>(parser, rng_type, [](curandGenerator_t gen, float* data, size_t size) {
            return curandGenerateNormal(gen, data, size, 0.0f, 1.0f);
        });
    }
    if(distribution == "normal-double")
    {
        run_benchmark<double>(
            parser, rng_type, [](curandGenerator_t gen, double* data, size_t size) {
                return curandGenerateNormalDouble(gen, data, size, 0.0, 1.0);
            });
    }
    if(distribution == "log-normal-float")
    {
        run_benchmark<float>(parser, rng_type, [](curandGenerator_t gen, float* data, size_t size) {
            return curandGenerateLogNormal(gen, data, size, 0.0f, 1.0f);
        });
    }
    if(distribution == "log-normal-double")
    {
        run_benchmark<double>(
            parser, rng_type, [](curandGenerator_t gen, double* data, size_t size) {
                return curandGenerateLogNormalDouble(gen, data, size, 0.0, 1.0);
            });
    }
    if(distribution == "poisson")
    {
        const auto lambdas = parser.get<std::vector<double>>("lambda");
        for(double lambda : lambdas)
        {
            std::cout << "    "
                      << "lambda " << std::fixed << std::setprecision(1) << lambda << std::endl;
            run_benchmark<unsigned int>(
                parser, rng_type, [lambda](curandGenerator_t gen, unsigned int* data, size_t size) {
                    return curandGeneratePoisson(gen, data, size, lambda);
                });
        }
    }
}

const std::vector<std::string> all_engines = {
    "xorwow",
    "mrg32k3a",
    "mtgp32",
    // "mt19937",
    "philox",
    "sobol32",
    // "scrambled_sobol32",
    // "sobol64",
    // "scrambled_sobol64",
};

const std::vector<std::string> all_distributions = {"uniform-uint",
                                                    "uniform-long-long",
                                                    "uniform-float",
                                                    "uniform-double",
                                                    "normal-float",
                                                    "normal-double",
                                                    "log-normal-float",
                                                    "log-normal-double",
                                                    "poisson"};

int main(int argc, char* argv[])
{
    cli::Parser parser(argc, argv);

    const std::string distribution_desc
        = "space-separated list of distributions:"
          + std::accumulate(all_distributions.begin(),
                            all_distributions.end(),
                            std::string(),
                            [](std::string a, std::string b) { return a + "\n      " + b; })
          + "\n      or all";
    const std::string engine_desc
        = "space-separated list of random number engines:"
          + std::accumulate(all_engines.begin(),
                            all_engines.end(),
                            std::string(),
                            [](std::string a, std::string b) { return a + "\n      " + b; })
          + "\n      or all";

    parser.set_optional<size_t>("size", "size", DEFAULT_RAND_N, "number of values");
    parser.set_optional<size_t>(
        "dimensions", "dimensions", 1, "number of dimensions of quasi-random values");
    parser.set_optional<size_t>("trials", "trials", 20, "number of trials");
    parser.set_optional<std::vector<std::string>>(
        "dis", "dis", {"uniform-uint"}, distribution_desc.c_str());
    parser.set_optional<std::vector<std::string>>(
        "engine", "engine", {"philox"}, engine_desc.c_str());
    parser.set_optional<std::vector<double>>(
        "lambda", "lambda", {10.0}, "space-separated list of lambdas of Poisson distribution");
    parser.run_and_exit_if_error();

    std::vector<std::string> engines;
    {
        auto es = parser.get<std::vector<std::string>>("engine");
        if(std::find(es.begin(), es.end(), "all") != es.end())
        {
            engines = all_engines;
        }
        else
        {
            for(auto e : all_engines)
            {
                if(std::find(es.begin(), es.end(), e) != es.end())
                    engines.push_back(e);
            }
        }
    }

    std::vector<std::string> distributions;
    {
        auto ds = parser.get<std::vector<std::string>>("dis");
        if(std::find(ds.begin(), ds.end(), "all") != ds.end())
        {
            distributions = all_distributions;
        }
        else
        {
            for(auto d : all_distributions)
            {
                if(std::find(ds.begin(), ds.end(), d) != ds.end())
                    distributions.push_back(d);
            }
        }
    }

    int version;
    CURAND_CALL(curandGetVersion(&version));
    int runtime_version;
    CUDA_CALL(cudaRuntimeGetVersion(&runtime_version));
    int device_id;
    CUDA_CALL(cudaGetDevice(&device_id));
    cudaDeviceProp props;
    CUDA_CALL(cudaGetDeviceProperties(&props, device_id));

    std::cout << "cuRAND: " << version << " ";
    std::cout << "Runtime: " << runtime_version << " ";
    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;

    for(auto engine : engines)
    {
        rng_type_t rng_type = CURAND_RNG_PSEUDO_XORWOW;
        if(engine == "xorwow")
            rng_type = CURAND_RNG_PSEUDO_XORWOW;
        else if(engine == "mrg32k3a")
            rng_type = CURAND_RNG_PSEUDO_MRG32K3A;
        else if(engine == "mtgp32")
            rng_type = CURAND_RNG_PSEUDO_MTGP32;
        else if(engine == "mt19937")
            rng_type = CURAND_RNG_PSEUDO_MT19937;
        else if(engine == "philox")
            rng_type = CURAND_RNG_PSEUDO_PHILOX4_32_10;
        else if(engine == "sobol32")
            rng_type = CURAND_RNG_QUASI_SOBOL32;
        else if(engine == "scrambled_sobol32")
            rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL32;
        else if(engine == "sobol64")
            rng_type = CURAND_RNG_QUASI_SOBOL64;
        else if(engine == "scrambled_sobol64")
            rng_type = CURAND_RNG_QUASI_SCRAMBLED_SOBOL64;
        else
        {
            std::cout << "Wrong engine name" << std::endl;
            exit(1);
        }

        std::cout << engine << ":" << std::endl;

        for(auto distribution : distributions)
        {
            std::cout << "  " << distribution << ":" << std::endl;
            run_benchmarks(parser, rng_type, distribution);
        }
        std::cout << std::endl;
    }

    return 0;
}
