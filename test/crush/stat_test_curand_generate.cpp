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

#include <boost/program_options.hpp>

#include <cuda_runtime.h>
#include <curand.h>

#include "stat_test_common.hpp"

extern "C" {
#include "gofs.h"
#include "fdist.h"
#include "fbar.h"
#include "finv.h"
}

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return exit(EXIT_FAILURE);}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return exit(EXIT_FAILURE);}} while(0)

typedef curandRngType rng_type_t;

template<typename T>
using generate_func_type = std::function<curandStatus_t(curandGenerator_t, T *, size_t)>;

template<typename T>
void run_test(const boost::program_options::variables_map& vm,
              const rng_type_t rng_type,
              const std::string plot_name,
              generate_func_type<T> generate_func,
              const double mean, const double stddev,
              distribution_func_type distribution_func)
{
    const size_t size = vm["size"].as<size_t>();
    const size_t level1_tests = vm["level1-tests"].as<size_t>();
    const size_t level2_tests = vm["level2-tests"].as<size_t>();
    const bool save_plots = vm.count("plots");

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

void run_tests(const boost::program_options::variables_map& vm,
               const rng_type_t rng_type,
               const std::string& distribution,
               const std::string plot_name)
{
    if (distribution == "uniform-float")
    {
        run_test<float>(vm, rng_type, plot_name,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateUniform(gen, data, size);
            },
            0.5, std::sqrt(1.0 / 12.0),
            [](double x) { return fdist_Unif(x); }
        );
    }
    if (distribution == "uniform-double")
    {
        run_test<double>(vm, rng_type, plot_name,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateUniformDouble(gen, data, size);
            },
            0.5, std::sqrt(1.0 / 12.0),
            [](double x) { return fdist_Unif(x); }
        );
    }
    if (distribution == "normal-float")
    {
        run_test<float>(vm, rng_type, plot_name,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateNormal(gen, data, size, 0.0f, 1.0f);
            },
            0.0, 1.0,
            [](double x) { return fdist_Normal2(x); }
        );
    }
    if (distribution == "normal-double")
    {
        run_test<double>(vm, rng_type, plot_name,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateNormalDouble(gen, data, size, 0.0, 1.0);
            },
            0.0, 1.0,
            [](double x) { return fdist_Normal2(x); }
        );
    }
    if (distribution == "log-normal-float")
    {
        run_test<float>(vm, rng_type, plot_name,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateLogNormal(gen, data, size, 0.0f, 1.0f);
            },
            std::exp(0.5), std::sqrt((std::exp(1.0) - 1.0) * std::exp(1.0)),
            [](double x) { return fdist_LogNormal(0.0, 1.0, x); }
        );
    }
    if (distribution == "log-normal-double")
    {
        run_test<double>(vm, rng_type, plot_name,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateLogNormalDouble(gen, data, size, 0.0, 1.0);
            },
            std::exp(0.5), std::sqrt((std::exp(1.0) - 1.0) * std::exp(1.0)),
            [](double x) { return fdist_LogNormal(0.0, 1.0, x); }
        );
    }
    if (distribution == "poisson")
    {
        const auto lambdas = vm["lambda"].as<std::vector<double>>();
        for (double lambda : lambdas)
        {
            std::cout << "    " << "lambda "
                 << std::fixed << std::setprecision(1) << lambda << std::endl;
            run_test<unsigned int>(vm, rng_type, plot_name + "-" + std::to_string(lambda),
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
    // "mtgp32",
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
    namespace po = boost::program_options;
    po::options_description options("options");

    const std::string distribution_desc =
        "space-separated list of distributions:" +
        std::accumulate(all_distributions.begin(), all_distributions.end(), std::string(),
            [](std::string a, std::string b) {
                return a + "\n   " + b;
            }
        ) +
        "\nor all";
    const std::string engine_desc =
        "space-separated list of random number engines:" +
        std::accumulate(all_engines.begin(), all_engines.end(), std::string(),
            [](std::string a, std::string b) {
                return a + "\n   " + b;
            }
        ) +
        "\nor all";
    options.add_options()
        ("help", "show usage instructions")
        ("size", po::value<size_t>()->default_value(10000), "number of samples in every first level test")
        ("level1-tests", po::value<size_t>()->default_value(10), "number of first level tests")
        ("level2-tests", po::value<size_t>()->default_value(10), "number of second level tests")
        ("dis", po::value<std::vector<std::string>>()->multitoken()->default_value({ "all" }, "all"),
            distribution_desc.c_str())
        ("engine", po::value<std::vector<std::string>>()->multitoken()->default_value({ "philox" }, "philox"),
            engine_desc.c_str())
        ("lambda", po::value<std::vector<double>>()->multitoken()->default_value({ 100.0 }, "100.0"),
            "space-separated list of lambdas of Poisson distribution")
        ("plots", "save plots for GnuPlot")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, options), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << options << std::endl;
        return 0;
    }

    std::vector<std::string> engines;
    {
        auto es = vm["engine"].as<std::vector<std::string>>();
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
        auto ds = vm["dis"].as<std::vector<std::string>>();
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

    std::cout << "cuRAND:" << std::endl << std::endl;
    for (auto engine : engines)
    {
        std::cout << engine << ":" << std::endl;
        for (auto distribution : distributions)
        {
            std::cout << "  " << distribution << ":" << std::endl;
            const std::string plot_name = engine + "-" + distribution;
            if (engine == "xorwow")
            {
                run_tests(vm, CURAND_RNG_PSEUDO_XORWOW, distribution, plot_name);
            }
            else if (engine == "mrg32k3a")
            {
                run_tests(vm, CURAND_RNG_PSEUDO_MRG32K3A, distribution, plot_name);
            }
            else if (engine == "philox")
            {
                run_tests(vm, CURAND_RNG_PSEUDO_PHILOX4_32_10, distribution, plot_name);
            }
            else if (engine == "sobol32")
            {
                run_tests(vm, CURAND_RNG_QUASI_SOBOL32, distribution, plot_name);
            }
        }
    }

    return 0;
}
