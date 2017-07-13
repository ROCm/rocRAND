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

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <numeric>
#include <utility>

#include <boost/program_options.hpp>

#include <cuda_runtime.h>
#include <curand.h>

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return exit(EXIT_FAILURE);}} while(0)

#ifndef DEFAULT_RAND_N
const size_t DEFAULT_RAND_N = 1024 * 1024 * 128;
#endif

template<typename T>
using generate_func_type = std::function<curandStatus_t(curandGenerator_t, T *, size_t)>;

template<typename T>
void run_benchmark(const size_t size, const size_t trials,
                   const curandRngType rng_type,
                   generate_func_type<T> generate_func)
{
    T * data;
    CUDA_CALL(cudaMalloc((void **)&data, size * sizeof(T)));

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, rng_type));
    CUDA_CALL(cudaDeviceSynchronize());

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

    std::cout << std::fixed << std::setprecision(3)
              << "     "
              << "Throughput = "
              << std::setw(8) << (trials * size * sizeof(T)) /
                    (elapsed.count() / 1e3 * (1 << 30))
              << " GB/s, AvgTime (1 trial) = "
              << std::setw(8) << elapsed.count() / trials
              << " ms, Time (all) = "
              << std::setw(8) << elapsed.count()
              << " ms, Size = " << size
              << std::endl;

    CURAND_CALL(curandDestroyGenerator(generator));
    CUDA_CALL(cudaFree(data));
}

void run_benchmarks(const size_t size, const size_t trials,
                    const curandRngType rng_type,
                    const std::string& distribution,
                    const boost::program_options::variables_map& vm)
{
    bool all = distribution == "all";
    if (distribution == "uniform-uint" || all)
    {
        if (rng_type != CURAND_RNG_QUASI_SOBOL64 &&
            rng_type != CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
        {
            std::cout << "  " << "uniform-uint:" << std::endl;
            run_benchmark<unsigned int>(size, trials, rng_type,
                [](curandGenerator_t gen, unsigned int * data, size_t size) {
                    return curandGenerate(gen, data, size);
                }
            );
        }
    }
    if (distribution == "uniform-long-long" || all)
    {
        if (rng_type == CURAND_RNG_QUASI_SOBOL64 ||
            rng_type == CURAND_RNG_QUASI_SCRAMBLED_SOBOL64)
        {
            std::cout << "  " << "uniform-long-long:" << std::endl;
            run_benchmark<unsigned long long>(size, trials, rng_type,
                [](curandGenerator_t gen, unsigned long long * data, size_t size) {
                    return curandGenerateLongLong(gen, data, size);
                }
            );
        }
    }
    if (distribution == "uniform-float" || all)
    {
        std::cout << "  " << "uniform-float:" << std::endl;
        run_benchmark<float>(size, trials, rng_type,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateUniform(gen, data, size);
            }
        );
    }
    if (distribution == "uniform-double" || all)
    {
        std::cout << "  " << "uniform-double:" << std::endl;
        run_benchmark<double>(size, trials, rng_type,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateUniformDouble(gen, data, size);
            }
        );
    }
    if (distribution == "normal-float" || all)
    {
        std::cout << "  " << "normal-float:" << std::endl;
        run_benchmark<float>(size, trials, rng_type,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateNormal(gen, data, size, 0.0f, 1.0f);
            }
        );
    }
    if (distribution == "normal-double" || all)
    {
        std::cout << "  " << "normal-double:" << std::endl;
        run_benchmark<double>(size, trials, rng_type,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateNormalDouble(gen, data, size, 0.0, 1.0);
            }
        );
    }
    if (distribution == "log-normal-float" || all)
    {
        std::cout << "  " << "log-normal-float:" << std::endl;
        run_benchmark<float>(size, trials, rng_type,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateLogNormal(gen, data, size, 0.0f, 1.0f);
            }
        );
    }
    if (distribution == "log-normal-double" || all)
    {
        std::cout << "  " << "log-normal-double:" << std::endl;
        run_benchmark<double>(size, trials, rng_type,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateLogNormalDouble(gen, data, size, 0.0, 1.0);
            }
        );
    }
    if (distribution == "poisson" || all)
    {
        const double lambda = vm["lambda"].as<double>();
        std::cout << "  " << "poisson (" << std::setprecision(1) << lambda << "):" << std::endl;
        run_benchmark<unsigned int>(size, trials, rng_type,
            [lambda](curandGenerator_t gen, unsigned int * data, size_t size) {
                return curandGeneratePoisson(gen, data, size, lambda);
            }
        );
    }
}

const std::vector<std::pair<curandRngType, std::string>> engines = {
    { CURAND_RNG_PSEUDO_XORWOW, "xorwow" },
    { CURAND_RNG_PSEUDO_MRG32K3A, "mrg32k3a" },
    { CURAND_RNG_PSEUDO_MTGP32, "mtgp32" },
    { CURAND_RNG_PSEUDO_MT19937, "mt19937" },
    { CURAND_RNG_PSEUDO_PHILOX4_32_10, "philox" },
    { CURAND_RNG_QUASI_SOBOL32, "sobol32" },
    { CURAND_RNG_QUASI_SCRAMBLED_SOBOL32, "scrambled_sobol32" },
    { CURAND_RNG_QUASI_SOBOL64, "sobol64" },
    { CURAND_RNG_QUASI_SCRAMBLED_SOBOL64, "scrambled_sobol64" }
};

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;
    po::options_description options("options");

    const std::string distribution_desc =
        "space-separated list of distributions:"
            "\n   uniform-uint"
            "\n   uniform-long-long"
            "\n   uniform-float"
            "\n   uniform-double"
            "\n   normal-float"
            "\n   normal-double"
            "\n   log-normal-float"
            "\n   log-normal-double"
            "\n   poisson"
            "\nor all";
    const std::string engine_desc =
        "space-separated list of random number engines:" +
        std::accumulate(engines.begin(), engines.end(), std::string(),
            [](std::string a, std::pair<curandRngType, std::string> b) {
                return a + "\n   " + b.second;
            }
        ) +
        "\nor all";
    options.add_options()
        ("help", "show usage instructions")
        ("size", po::value<size_t>()->default_value(DEFAULT_RAND_N), "number of values")
        ("trials", po::value<size_t>()->default_value(20), "number of trials")
        ("dis", po::value<std::vector<std::string>>()->multitoken()->default_value({ "uniform-uint" }, "uniform-uint"),
            distribution_desc.c_str())
        ("engine", po::value<std::vector<std::string>>()->multitoken()->default_value({ "philox" }, "philox"),
            engine_desc.c_str())
        ("lambda", po::value<double>()->default_value(100.0), "lambda of Poisson distribution")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cout << options << std::endl;
        return 0;
    }

    const size_t size = vm["size"].as<size_t>();
    const size_t trials = vm["trials"].as<size_t>();

    std::cout << "cuRAND:" << std::endl;

    for (auto engine : vm["engine"].as<std::vector<std::string>>())
    for (auto e : engines)
    {
        if (engine == e.second || engine == "all")
        {
            std::cout << std::endl << e.second << ":" << std::endl;
            const curandRngType rng_type = e.first;
            for (auto distribution : vm["dis"].as<std::vector<std::string>>())
            {
                run_benchmarks(size, trials, rng_type, distribution, vm);
            }
        }
    }

    return 0;
}
