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
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <utility>
#include <type_traits>
#include <algorithm>

#include <cuda_runtime.h>
#include <curand.h>

#include "stat_test_common.hpp"
#include "cmdparser.hpp"

extern "C" {
#include "gofs.h"
#include "fdist.h"
#include "fbar.h"
#include "finv.h"
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

typedef curandRngType rng_type_t;

template<typename T>
using generate_func_type = std::function<curandStatus_t(curandGenerator_t, T *, size_t)>;

template<typename T>
void run_test(const cli::Parser& parser,
              const rng_type_t rng_type,
              const std::string plot_name,
              generate_func_type<T> generate_func,
              const double mean, const double stddev,
              distribution_func_type distribution_func)
{
    const size_t size = parser.get<size_t>("size");
    const size_t level1_tests = parser.get<size_t>("level1-tests");
    const size_t level2_tests = parser.get<size_t>("level2-tests");
    const bool save_plots = parser.get<bool>("plots");

    T * data;
    CUDA_CALL(cudaMalloc((void **)&data, size * level1_tests * sizeof(T)));

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, rng_type));

    const size_t dimensions = level1_tests;
    curandStatus_t status = curandSetQuasiRandomGeneratorDimensions(generator, dimensions);
    if (status != CURAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
    {
        CURAND_CALL(status);
    }

    for (size_t level2_test = 0; level2_test < level2_tests; level2_test++)
    {
        CURAND_CALL(generate_func(generator, data, size * level1_tests));
        CUDA_CALL(cudaDeviceSynchronize());

        std::vector<T> h_data(size * level1_tests);
        CUDA_CALL(cudaMemcpy(h_data.data(), data, size * level1_tests * sizeof(T), cudaMemcpyDeviceToHost));

        analyze(size, level1_tests, h_data.data(),
                save_plots, plot_name + "-" + std::to_string(level2_test),
                mean, stddev, distribution_func);
    }

    CURAND_CALL(curandDestroyGenerator(generator));
    CUDA_CALL(cudaFree(data));
}

void run_tests(const cli::Parser& parser,
               const rng_type_t rng_type,
               const std::string& distribution,
               const std::string plot_name)
{
    if (distribution == "uniform-float")
    {
        run_test<float>(parser, rng_type, plot_name,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateUniform(gen, data, size);
            },
            0.5, std::sqrt(1.0 / 12.0),
            [](double x) { return fdist_Unif(x); }
        );
    }
    if (distribution == "uniform-double")
    {
        run_test<double>(parser, rng_type, plot_name,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateUniformDouble(gen, data, size);
            },
            0.5, std::sqrt(1.0 / 12.0),
            [](double x) { return fdist_Unif(x); }
        );
    }
    if (distribution == "normal-float")
    {
        run_test<float>(parser, rng_type, plot_name,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateNormal(gen, data, size, 0.0f, 1.0f);
            },
            0.0, 1.0,
            [](double x) { return fdist_Normal2(x); }
        );
    }
    if (distribution == "normal-double")
    {
        run_test<double>(parser, rng_type, plot_name,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateNormalDouble(gen, data, size, 0.0, 1.0);
            },
            0.0, 1.0,
            [](double x) { return fdist_Normal2(x); }
        );
    }
    if (distribution == "log-normal-float")
    {
        run_test<float>(parser, rng_type, plot_name,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateLogNormal(gen, data, size, 0.0f, 1.0f);
            },
            std::exp(0.5), std::sqrt((std::exp(1.0) - 1.0) * std::exp(1.0)),
            [](double x) { return fdist_LogNormal(0.0, 1.0, x); }
        );
    }
    if (distribution == "log-normal-double")
    {
        run_test<double>(parser, rng_type, plot_name,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateLogNormalDouble(gen, data, size, 0.0, 1.0);
            },
            std::exp(0.5), std::sqrt((std::exp(1.0) - 1.0) * std::exp(1.0)),
            [](double x) { return fdist_LogNormal(0.0, 1.0, x); }
        );
    }
    if (distribution == "poisson")
    {
        const auto lambdas = parser.get<std::vector<double>>("lambda");
        for (double lambda : lambdas)
        {
            std::cout << "    " << "lambda "
                 << std::fixed << std::setprecision(1) << lambda << std::endl;
            run_test<unsigned int>(parser, rng_type, plot_name + "-" + std::to_string(lambda),
                [lambda](curandGenerator_t gen, unsigned int * data, size_t size) {
                    return curandGeneratePoisson(gen, data, size, lambda);
                },
                lambda, std::sqrt(lambda),
                [lambda](double x) { return fdist_Poisson1(lambda, static_cast<long>(std::round(x)) - 1); }
            );
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

const std::vector<std::string> all_distributions = {
    "uniform-float",
    "uniform-double",
    "normal-float",
    "normal-double",
    "log-normal-float",
    "log-normal-double",
    "poisson",
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

    parser.set_optional<size_t>("size", "size", 10000, "number of samples in every first level test");
    parser.set_optional<size_t>("level1-tests", "level1-tests", 10, "number of first level tests");
    parser.set_optional<size_t>("level2-tests", "level2-tests", 10, "number of second level tests");
    parser.set_optional<std::vector<std::string>>("dis", "dis", {"all"}, distribution_desc.c_str());
    parser.set_optional<std::vector<std::string>>("engine", "engine", {"philox"}, engine_desc.c_str());
    parser.set_optional<std::vector<double>>("lambda", "lambda", {100.0}, "space-separated list of lambdas of Poisson distribution");
    parser.set_optional<bool>("plots", "plots", false, "Boolean argument to save plots for GnuPlot");
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

    for (auto engine : engines)
    {
        std::cout << engine << ":" << std::endl;
        for (auto distribution : distributions)
        {
            std::cout << "  " << distribution << ":" << std::endl;
            const std::string plot_name = engine + "-" + distribution;
            if (engine == "xorwow")
            {
                run_tests(parser, CURAND_RNG_PSEUDO_XORWOW, distribution, plot_name);
            }
            else if (engine == "mrg32k3a")
            {
                run_tests(parser, CURAND_RNG_PSEUDO_MRG32K3A, distribution, plot_name);
            }
            else if (engine == "mtgp32")
            {
                run_tests(parser, CURAND_RNG_PSEUDO_MTGP32, distribution, plot_name);
            }
            else if (engine == "philox")
            {
                run_tests(parser, CURAND_RNG_PSEUDO_PHILOX4_32_10, distribution, plot_name);
            }
            else if (engine == "sobol32")
            {
                run_tests(parser, CURAND_RNG_QUASI_SOBOL32, distribution, plot_name);
            }
        }
    }

    return 0;
}
