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

#include <hip/hip_runtime.h>
#include <rocrand.h>

#include "stat_test_common.hpp"
#include "cmdparser.hpp"

extern "C" {
#include "gofs.h"
#include "fdist.h"
#include "fbar.h"
#include "finv.h"
}

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

typedef rocrand_rng_type rng_type_t;

template<typename T>
using generate_func_type = std::function<rocrand_status(rocrand_generator, T *, size_t)>;

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
    HIP_CHECK(hipMalloc((void **)&data, size * level1_tests * sizeof(T)));

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));

    const size_t dimensions = level1_tests;
    rocrand_status status = rocrand_set_quasi_random_generator_dimensions(generator, dimensions);
    if (status != ROCRAND_STATUS_TYPE_ERROR) // If the RNG is not quasi-random
    {
        ROCRAND_CHECK(status);
    }

    for (size_t level2_test = 0; level2_test < level2_tests; level2_test++)
    {
        ROCRAND_CHECK(generate_func(generator, data, size * level1_tests));
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<T> h_data(size * level1_tests);
        HIP_CHECK(hipMemcpy(h_data.data(), data, size * level1_tests * sizeof(T), hipMemcpyDeviceToHost));

        analyze(size, level1_tests, h_data.data(),
                save_plots, plot_name + "-" + std::to_string(level2_test),
                mean, stddev, distribution_func);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    HIP_CHECK(hipFree(data));
}

void run_tests(const cli::Parser& parser,
               const rng_type_t rng_type,
               const std::string& distribution,
               const std::string plot_name)
{
    if (distribution == "uniform-float")
    {
        run_test<float>(parser, rng_type, plot_name,
            [](rocrand_generator gen, float * data, size_t size) {
                return rocrand_generate_uniform(gen, data, size);
            },
            0.5, std::sqrt(1.0 / 12.0),
            [](double x) { return fdist_Unif(x); }
        );
    }
    if (distribution == "uniform-double")
    {
        run_test<double>(parser, rng_type, plot_name,
            [](rocrand_generator gen, double * data, size_t size) {
                return rocrand_generate_uniform_double(gen, data, size);
            },
            0.5, std::sqrt(1.0 / 12.0),
            [](double x) { return fdist_Unif(x); }
        );
    }
    if (distribution == "normal-float")
    {
        run_test<float>(parser, rng_type, plot_name,
            [](rocrand_generator gen, float * data, size_t size) {
                return rocrand_generate_normal(gen, data, size, 0.0f, 1.0f);
            },
            0.0, 1.0,
            [](double x) { return fdist_Normal2(x); }
        );
    }
    if (distribution == "normal-double")
    {
        run_test<double>(parser, rng_type, plot_name,
            [](rocrand_generator gen, double * data, size_t size) {
                return rocrand_generate_normal_double(gen, data, size, 0.0, 1.0);
            },
            0.0, 1.0,
            [](double x) { return fdist_Normal2(x); }
        );
    }
    if (distribution == "log-normal-float")
    {
        run_test<float>(parser, rng_type, plot_name,
            [](rocrand_generator gen, float * data, size_t size) {
                return rocrand_generate_log_normal(gen, data, size, 0.0f, 1.0f);
            },
            std::exp(0.5), std::sqrt((std::exp(1.0) - 1.0) * std::exp(1.0)),
            [](double x) { return fdist_LogNormal(0.0, 1.0, x); }
        );
    }
    if (distribution == "log-normal-double")
    {
        run_test<double>(parser, rng_type, plot_name,
            [](rocrand_generator gen, double * data, size_t size) {
                return rocrand_generate_log_normal_double(gen, data, size, 0.0, 1.0);
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
                [lambda](rocrand_generator gen, unsigned int * data, size_t size) {
                    return rocrand_generate_poisson(gen, data, size, lambda);
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
        std::cout << engine << ":" << std::endl;
        for (auto distribution : distributions)
        {
            std::cout << "  " << distribution << ":" << std::endl;
            const std::string plot_name = engine + "-" + distribution;
            if (engine == "xorwow")
            {
                run_tests(parser, ROCRAND_RNG_PSEUDO_XORWOW, distribution, plot_name);
            }
            else if (engine == "mrg32k3a")
            {
                run_tests(parser, ROCRAND_RNG_PSEUDO_MRG32K3A, distribution, plot_name);
            }
            else if (engine == "philox")
            {
                run_tests(parser, ROCRAND_RNG_PSEUDO_PHILOX4_32_10, distribution, plot_name);
            }
            else if (engine == "sobol32")
            {
                run_tests(parser, ROCRAND_RNG_QUASI_SOBOL32, distribution, plot_name);
            }
            else if (engine == "mtgp32")
            {
                run_tests(parser, ROCRAND_RNG_PSEUDO_MTGP32, distribution, plot_name);
            }
        }
    }

    return 0;
}
