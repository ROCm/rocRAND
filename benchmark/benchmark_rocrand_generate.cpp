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

#include <hip/hip_runtime.h>
#include <rocrand.h>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << error << std::endl; \
        exit(error); \
    } \
  }

#define ROCRAND_CHECK(condition)                 \
  {                                              \
    rocrand_status status = condition;           \
    if(status != ROCRAND_STATUS_SUCCESS) {       \
        std::cout << status << std::endl; \
        exit(status); \
    } \
  }

#ifndef DEFAULT_RAND_N
const size_t DEFAULT_RAND_N = 1024 * 1024 * 128;
#endif

template<typename T>
using generate_func_type = std::function<rocrand_status(rocrand_generator, T *, size_t)>;

template<typename T>
void run_benchmark(const size_t size, const size_t trials,
                   const rocrand_rng_type rng_type,
                   generate_func_type<T> generate_func)
{
    T * data;
    HIP_CHECK(hipMalloc((void **)&data, size * sizeof(T)));

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));
    // Make sure memory is allocated
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for (size_t i = 0; i < 5; i++)
    {
        ROCRAND_CHECK(generate_func(generator, data, size));
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Measurement
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < trials; i++)
    {
        ROCRAND_CHECK(generate_func(generator, data, size));
    }
    HIP_CHECK(hipDeviceSynchronize());
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

    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    HIP_CHECK(hipFree(data));
}

void run_benchmarks(const size_t size, const size_t trials,
                    const rocrand_rng_type rng_type,
                    const std::string& distribution,
                    const boost::program_options::variables_map& vm)
{
    bool all = distribution == "all";
    if (distribution == "uniform-uint" || all)
    {
        std::cout << "  " << "uniform-uint:" << std::endl;
        run_benchmark<unsigned int>(size, trials, rng_type,
            [](rocrand_generator gen, unsigned int * data, size_t size) {
                return rocrand_generate(gen, data, size);
            }
        );
    }
    if (distribution == "uniform-float" || all)
    {
        std::cout << "  " << "uniform-float:" << std::endl;
        run_benchmark<float>(size, trials, rng_type,
            [](rocrand_generator gen, float * data, size_t size) {
                return rocrand_generate_uniform(gen, data, size);
            }
        );
    }
    if (distribution == "uniform-double" || all)
    {
        std::cout << "  " << "uniform-double:" << std::endl;
        run_benchmark<double>(size, trials, rng_type,
            [](rocrand_generator gen, double * data, size_t size) {
                return rocrand_generate_uniform_double(gen, data, size);
            }
        );
    }
    if (distribution == "normal-float" || all)
    {
        std::cout << "  " << "normal-float:" << std::endl;
        run_benchmark<float>(size, trials, rng_type,
            [](rocrand_generator gen, float * data, size_t size) {
                return rocrand_generate_normal(gen, data, size, 0.0f, 1.0);
            }
        );
    }
    if (distribution == "normal-double" || all)
    {
        std::cout << "  " << "normal-double:" << std::endl;
        run_benchmark<double>(size, trials, rng_type,
            [](rocrand_generator gen, double * data, size_t size) {
                return rocrand_generate_normal_double(gen, data, size, 0.0f, 1.0f);
            }
        );
    }
    if (distribution == "log-normal-float" || all)
    {
        std::cout << "  " << "log-normal-float:" << std::endl;
        run_benchmark<float>(size, trials, rng_type,
            [](rocrand_generator gen, float * data, size_t size) {
                return rocrand_generate_log_normal(gen, data, size, 0.0f, 1.0);
            }
        );
    }
    if (distribution == "log-normal-double" || all)
    {
        std::cout << "  " << "log-normal-double:" << std::endl;
        run_benchmark<double>(size, trials, rng_type,
            [](rocrand_generator gen, double * data, size_t size) {
                return rocrand_generate_log_normal_double(gen, data, size, 0.0f, 1.0f);
            }
        );
    }
    if (distribution == "poisson" || all)
    {
        std::cout << "  " << "poisson:" << std::endl;
        const double lambda = vm["lambda"].as<double>();
        run_benchmark<unsigned int>(size, trials, rng_type,
            [lambda](rocrand_generator gen, unsigned int * data, size_t size) {
                return rocrand_generate_poisson(gen, data, size, lambda);
            }
        );
    }
}

const std::vector<std::pair<rocrand_rng_type, std::string>> engines = {
    // { ROCRAND_RNG_PSEUDO_XORWOW, "xorwow" },
    { ROCRAND_RNG_PSEUDO_MRG32K3A, "mrg32k3a" },
    // { ROCRAND_RNG_PSEUDO_MTGP32, "mtgp32" },
    { ROCRAND_RNG_PSEUDO_PHILOX4_32_10, "philox" },
    // { ROCRAND_RNG_QUASI_SOBOL32, "sobol32" },
};

int main(int argc, char *argv[])
{
    namespace po = boost::program_options;
    po::options_description options("options");

    const std::string distribution_desc =
        "space-separated list of distributions:"
            "\n   uniform-uint"
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
            [](std::string a, std::pair<rocrand_rng_type, std::string> b) {
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

    if(vm.count("help")) {
        std::cout << options << std::endl;
        return 0;
    }

    const size_t size = vm["size"].as<size_t>();
    const size_t trials = vm["trials"].as<size_t>();

    std::cout << "rocRAND:" << std::endl;
    for (auto engine : vm["engine"].as<std::vector<std::string>>())
    for (auto e : engines)
    {
        if (engine == e.second || engine == "all")
        {
            std::cout << std::endl << e.second << ":" << std::endl;
            const rocrand_rng_type rng_type = e.first;
            for (auto distribution : vm["dis"].as<std::vector<std::string>>())
            {
                run_benchmarks(size, trials, rng_type, distribution, vm);
            }
        }
    }

    return 0;
}
