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
#include <algorithm>

#include "cmdparser.hpp"

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#define HIP_CHECK(condition)         \
  {                                  \
    hipError_t error = condition;    \
    if(error != hipSuccess){         \
        std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl; \
        exit(error); \
    } \
  }

#define ROCRAND_CHECK(condition)                 \
  {                                              \
    rocrand_status _status = condition;           \
    if(_status != ROCRAND_STATUS_SUCCESS) {       \
        std::cout << "ROCRAND error: " << _status << " line: " << __LINE__ << std::endl; \
        exit(_status); \
    } \
  }

#ifndef DEFAULT_RAND_N
const size_t DEFAULT_RAND_N = 1024 * 1024 * 128;
#endif

typedef rocrand_rng_type rng_type_t;

template<typename T>
using generate_func_type = std::function<rocrand_status(rocrand_generator, T *, size_t)>;

template<typename T>
void run_benchmark(const cli::Parser& parser,
                   const rng_type_t rng_type,
                   generate_func_type<T> generate_func)
{
    const size_t size0 = parser.get<size_t>("size");
    const size_t trials = parser.get<size_t>("trials");
    const size_t dimensions = parser.get<size_t>("dimensions");
    const size_t size = (size0 / dimensions) * dimensions;

    T * data;
    HIP_CHECK(hipMalloc((void **)&data, size * sizeof(T)));

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

    rocrand_status status = rocrand_set_quasi_random_generator_dimensions(generator, dimensions);
    if (status != ROCRAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
    {
        ROCRAND_CHECK(status);
    }

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
              << "      "
              << "Throughput = "
              << std::setw(8) << (trials * size * sizeof(T)) /
                    (elapsed.count() / 1e3 * (1 << 30))
              << " GB/s, Samples = "
              << std::setw(8) << (trials * size) /
                    (elapsed.count() / 1e3 * (1 << 30))
              << " GSample/s, AvgTime (1 trial) = "
              << std::setw(8) << elapsed.count() / trials
              << " ms, Time (all) = "
              << std::setw(8) << elapsed.count()
              << " ms, Size = " << size
              << std::endl;

    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    HIP_CHECK(hipFree(data));
}

void run_benchmarks(const cli::Parser& parser,
                    const rng_type_t rng_type,
                    const std::string& distribution)
{
    if (distribution == "uniform-uint")
    {
        run_benchmark<unsigned int>(parser, rng_type,
            [](rocrand_generator gen, unsigned int * data, size_t size) {
                return rocrand_generate(gen, data, size);
            }
        );
    }
    if (distribution == "uniform-uchar")
    {
        run_benchmark<unsigned char>(parser, rng_type,
            [](rocrand_generator gen, unsigned char * data, size_t size) {
                return rocrand_generate_char(gen, data, size);
            }
        );
    }
    if (distribution == "uniform-ushort")
    {
        run_benchmark<unsigned short>(parser, rng_type,
            [](rocrand_generator gen, unsigned short * data, size_t size) {
                return rocrand_generate_short(gen, data, size);
            }
        );
    }
    if (distribution == "uniform-half")
    {
        run_benchmark<__half>(parser, rng_type,
            [](rocrand_generator gen, __half * data, size_t size) {
                return rocrand_generate_uniform_half(gen, data, size);
            }
        );
    }
    if (distribution == "uniform-float")
    {
        run_benchmark<float>(parser, rng_type,
            [](rocrand_generator gen, float * data, size_t size) {
                return rocrand_generate_uniform(gen, data, size);
            }
        );
    }
    if (distribution == "uniform-double")
    {
        run_benchmark<double>(parser, rng_type,
            [](rocrand_generator gen, double * data, size_t size) {
                return rocrand_generate_uniform_double(gen, data, size);
            }
        );
    }
    if (distribution == "normal-half")
    {
        run_benchmark<__half>(parser, rng_type,
            [](rocrand_generator gen, __half * data, size_t size) {
                return rocrand_generate_normal_half(gen, data, size, 0.0f, 1.0f);
            }
        );
    }
    if (distribution == "normal-float")
    {
        run_benchmark<float>(parser, rng_type,
            [](rocrand_generator gen, float * data, size_t size) {
                return rocrand_generate_normal(gen, data, size, 0.0f, 1.0f);
            }
        );
    }
    if (distribution == "normal-double")
    {
        run_benchmark<double>(parser, rng_type,
            [](rocrand_generator gen, double * data, size_t size) {
                return rocrand_generate_normal_double(gen, data, size, 0.0, 1.0);
            }
        );
    }
    if (distribution == "log-normal-half")
    {
        run_benchmark<__half>(parser, rng_type,
            [](rocrand_generator gen, __half * data, size_t size) {
                return rocrand_generate_log_normal_half(gen, data, size, 0.0f, 1.0f);
            }
        );
    }
    if (distribution == "log-normal-float")
    {
        run_benchmark<float>(parser, rng_type,
            [](rocrand_generator gen, float * data, size_t size) {
                return rocrand_generate_log_normal(gen, data, size, 0.0f, 1.0f);
            }
        );
    }
    if (distribution == "log-normal-double")
    {
        run_benchmark<double>(parser, rng_type,
            [](rocrand_generator gen, double * data, size_t size) {
                return rocrand_generate_log_normal_double(gen, data, size, 0.0, 1.0);
            }
        );
    }
    if (distribution == "poisson")
    {
        const auto lambdas = parser.get<std::vector<double>>("lambda");
        for (double lambda : lambdas)
        {
            std::cout << "    " << "lambda "
                 << std::fixed << std::setprecision(1) << lambda << std::endl;
            run_benchmark<unsigned int>(parser, rng_type,
                [lambda](rocrand_generator gen, unsigned int * data, size_t size) {
                    return rocrand_generate_poisson(gen, data, size, lambda);
                }
            );
        }
    }
}

const std::vector<std::string> all_engines = {
    "xorwow",
    "mrg32k3a",
    "mtgp32",
    "philox",
    "sobol32",
};

const std::vector<std::string> all_distributions = {
    "uniform-uint",
    "uniform-uchar",
    "uniform-ushort",
    "uniform-half",
    // "uniform-long-long",
    "uniform-float",
    "uniform-double",
    "normal-half",
    "normal-float",
    "normal-double",
    "log-normal-half",
    "log-normal-float",
    "log-normal-double",
    "poisson"
};

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);

    const std::string distribution_desc =
        "space-separated list of distributions:" +
        std::accumulate(all_distributions.begin(), all_distributions.end(), std::string(),
            [](std::string a, std::string b) {
                return a + "\n      " + b;
            }
        ) +
        "\n      or all";
    const std::string engine_desc =
        "space-separated list of random number engines:" +
        std::accumulate(all_engines.begin(), all_engines.end(), std::string(),
            [](std::string a, std::string b) {
                return a + "\n      " + b;
            }
        ) +
        "\n      or all";

    parser.set_optional<size_t>("size", "size", DEFAULT_RAND_N, "number of values");
    parser.set_optional<size_t>("dimensions", "dimensions", 1, "number of dimensions of quasi-random values");
    parser.set_optional<size_t>("trials", "trials", 20, "number of trials");
    parser.set_optional<std::vector<std::string>>("dis", "dis", {"uniform-uint"}, distribution_desc.c_str());
    parser.set_optional<std::vector<std::string>>("engine", "engine", {"philox"}, engine_desc.c_str());
    parser.set_optional<std::vector<double>>("lambda", "lambda", {10.0}, "space-separated list of lambdas of Poisson distribution");
    parser.run_and_exit_if_error();

    std::vector<std::string> engines;
    {
        auto es = parser.get<std::vector<std::string>>("engine");
        if (std::find(es.begin(), es.end(), "all") != es.end())
        {
            engines = all_engines;
        }
        else
        {
            for (auto e : all_engines)
            {
                if (std::find(es.begin(), es.end(), e) != es.end())
                    engines.push_back(e);
            }
        }
    }

    std::vector<std::string> distributions;
    {
        auto ds = parser.get<std::vector<std::string>>("dis");
        if (std::find(ds.begin(), ds.end(), "all") != ds.end())
        {
            distributions = all_distributions;
        }
        else
        {
            for (auto d : all_distributions)
            {
                if (std::find(ds.begin(), ds.end(), d) != ds.end())
                    distributions.push_back(d);
            }
        }
    }

    int version;
    ROCRAND_CHECK(rocrand_get_version(&version));
    int runtime_version;
    HIP_CHECK(hipRuntimeGetVersion(&runtime_version));
    int device_id;
    HIP_CHECK(hipGetDevice(&device_id));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device_id));

    std::cout << "rocRAND: " << version << " ";
    std::cout << "Runtime: " << runtime_version << " ";
    std::cout << "Device: " << props.name;
    std::cout << std::endl << std::endl;

    for (auto engine : engines)
    {
        rng_type_t rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
        if (engine == "xorwow")
            rng_type = ROCRAND_RNG_PSEUDO_XORWOW;
        else if (engine == "mrg32k3a")
            rng_type = ROCRAND_RNG_PSEUDO_MRG32K3A;
        else if (engine == "philox")
            rng_type = ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
        else if (engine == "sobol32")
            rng_type = ROCRAND_RNG_QUASI_SOBOL32;
        else if (engine == "mtgp32")
            rng_type = ROCRAND_RNG_PSEUDO_MTGP32;
        else
        {
            std::cout << "Wrong engine name" << std::endl;
            exit(1);
        }

        std::cout << engine << ":" << std::endl;

        for (auto distribution : distributions)
        {
            std::cout << "  " << distribution << ":" << std::endl;
            run_benchmarks(parser, rng_type, distribution);
        }
        std::cout << std::endl;
    }

    return 0;
}
