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

template<typename T>
using generate_func_type = std::function<curandStatus_t(curandGenerator_t, T *, size_t)>;

using distribution_func_type = std::function<double(double)>;

template<typename T>
void run_test(const size_t size, const size_t trials,
              const curandRngType rng_type,
              const bool save_plots, const std::string plot_name,
              generate_func_type<T> generate_func,
              const double mean, const double stddev,
              distribution_func_type distribution_func)
{
    const std::vector<size_t> cells_counts({ 1000, 100, 25 });
    const double significance_level = 0.9;
    std::vector<double> rejection_criteria;
    for (size_t cells_count : cells_counts)
    {
        const double c = finv_ChiSquare2(static_cast<long>(cells_count), significance_level);
        rejection_criteria.push_back(c);
    }

    const int w = 14;

    // Header
    {
        std::cout << "  ";
        for (size_t cells_count : cells_counts)
        {
            std::cout << std::setw(w) << ("P" + std::to_string(cells_count));
            std::cout << " ";
        }
        std::cout << std::endl;
        std::cout << "  ";
        for (double c : rejection_criteria)
        {
            std::cout << std::setw(w) << ("< " + std::to_string(static_cast<int>(c)));
            std::cout << " ";
        }
        std::cout << std::endl << std::endl;
    }

    T * data;
    CUDA_CALL(cudaMalloc((void **)&data, size * sizeof(T)));

    curandGenerator_t generator;
    CURAND_CALL(curandCreateGenerator(&generator, rng_type));
    CUDA_CALL(cudaDeviceSynchronize());

    for (size_t trial = 0; trial < trials; trial++)
    {
        CURAND_CALL(generate_func(generator, data, size));
        CUDA_CALL(cudaDeviceSynchronize());

        std::vector<T> h_data(size);
        CUDA_CALL(cudaMemcpy(h_data.data(), data, size * sizeof(T), cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaDeviceSynchronize());

        std::cout << "  ";
        for (size_t test = 0; test < cells_counts.size(); test++)
        {
            const size_t cells_count = cells_counts[test];
            const double rejection_criterion = rejection_criteria[test];

            double start = (mean - 6.0 * stddev);
            double cell_width = 12.0 * stddev / cells_count;
            if (std::is_integral<T>::value)
            {
                // Use integral values for discrete distributions (e.g. Poisson)
                start = std::floor(start);
                cell_width = std::ceil(cell_width);
            }

            std::vector<unsigned int> historgram(cells_count);

            unsigned int count = 0;
            for (size_t si = 0; si < size; si++)
            {
                const double v = h_data[si];
                const int cell = static_cast<int>((v - start) / cell_width);
                if (cell >= 0 && cell < cells_count)
                {
                    historgram[cell]++;
                    count++;
                }
            }

            std::ofstream fout;
            if (save_plots)
            {
                fout.open(plot_name + "-" + std::to_string(trial) + "-" + std::to_string(test) + ".dat",
                    std::ios_base::out | std::ios_base::trunc);
            }
            double chi_squared = 0.0;
            for (size_t ci = 0; ci < cells_count; ci++)
            {
                const double x0 = start + ci * cell_width;
                const double x1 = start + (ci + 1) * cell_width;
                const double observed = historgram[ci] / static_cast<double>(count);
                const double expected = distribution_func(x1) - distribution_func(x0);
                if (expected > 0.0)
                {
                    chi_squared += (observed - expected) * (observed - expected) / expected;
                }
                if (save_plots)
                {
                    fout << observed << "\t" << expected << std::endl;
                }
            }
            chi_squared *= count;

            std::cout << std::setw(w) << std::fixed << std::setprecision(5) << chi_squared;
            std::cout << (chi_squared < rejection_criterion ? " " : "*");
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    CURAND_CALL(curandDestroyGenerator(generator));
    CUDA_CALL(cudaFree(data));
}

void run_tests(const size_t size, const size_t trials,
               const curandRngType rng_type,
               const std::string& distribution,
               const bool save_plots, const std::string plot_name,
               const boost::program_options::variables_map& vm)
{
    std::cout << "  " << distribution << ":" << std::endl;
    if (distribution == "uniform-float")
    {
        run_test<float>(size, trials, rng_type, save_plots, plot_name,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateUniform(gen, data, size);
            },
            0.5, std::sqrt(1.0 / 12.0),
            [](double x) { return fdist_Unif(x); }
        );
    }
    if (distribution == "uniform-double")
    {
        run_test<double>(size, trials, rng_type, save_plots, plot_name,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateUniformDouble(gen, data, size);
            },
            0.5, std::sqrt(1.0 / 12.0),
            [](double x) { return fdist_Unif(x); }
        );
    }
    if (distribution == "normal-float")
    {
        run_test<float>(size, trials, rng_type, save_plots, plot_name,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateNormal(gen, data, size, 0.0f, 1.0f);
            },
            0.0, 1.0,
            [](double x) { return fdist_Normal2(x); }
        );
    }
    if (distribution == "normal-double")
    {
        run_test<double>(size, trials, rng_type, save_plots, plot_name,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateNormalDouble(gen, data, size, 0.0, 1.0);
            },
            0.0, 1.0,
            [](double x) { return fdist_Normal2(x); }
        );
    }
    if (distribution == "log-normal-float")
    {
        run_test<float>(size, trials, rng_type, save_plots, plot_name,
            [](curandGenerator_t gen, float * data, size_t size) {
                return curandGenerateLogNormal(gen, data, size, 0.0f, 1.0f);
            },
            0.0, 1.0,
            [](double x) { return fdist_LogNormal(0.0, 1.0, x); }
        );
    }
    if (distribution == "log-normal-double")
    {
        run_test<double>(size, trials, rng_type, save_plots, plot_name,
            [](curandGenerator_t gen, double * data, size_t size) {
                return curandGenerateLogNormalDouble(gen, data, size, 0.0, 1.0);
            },
            0.0, 1.0,
            [](double x) { return fdist_LogNormal(0.0, 1.0, x); }
        );
    }
    if (distribution == "poisson")
    {
        const double lambda = vm["lambda"].as<double>();
        std::cout << "  " << "lambda:"
             << std::fixed << std::setprecision(1) << lambda << ":" << std::endl;
        run_test<unsigned int>(size, trials, rng_type, save_plots, plot_name,
            [lambda](curandGenerator_t gen, unsigned int * data, size_t size) {
                return curandGeneratePoisson(gen, data, size, lambda);
            },
            lambda, std::sqrt(lambda),
            [lambda](double x) { return fdist_Poisson1(lambda, static_cast<long>(std::round(x))); }
        );
    }
}

const std::vector<std::pair<curandRngType, std::string>> all_engines = {
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
            [](std::string a, std::pair<curandRngType, std::string> b) {
                return a + "\n   " + b.second;
            }
        ) +
        "\nor all";
    options.add_options()
        ("help", "show usage instructions")
        ("size", po::value<size_t>()->default_value(10000), "number of values")
        ("trials", po::value<size_t>()->default_value(20), "number of trials")
        ("dis", po::value<std::vector<std::string>>()->multitoken()->default_value({ "all" }, "all"),
            distribution_desc.c_str())
        ("engine", po::value<std::vector<std::string>>()->multitoken()->default_value({ "philox" }, "philox"),
            engine_desc.c_str())
        ("lambda", po::value<double>()->default_value(1000.0), "lambda of Poisson distribution")
        ("plots", "save plots for GnuPlot")
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
    const bool save_plots = vm.count("plots");

    std::cout << "cuRAND:" << std::endl << std::endl;
    std::vector<std::pair<curandRngType, std::string>> engines;
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
                if (std::find(es.begin(), es.end(), e.second) != es.end())
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

    for (auto e : engines)
    {
        const std::string engine_name = e.second;
        std::cout << engine_name << ":" << std::endl;
        const curandRngType rng_type = e.first;
        for (auto distribution : distributions)
        {
            const std::string plot_name = engine_name + "-" + distribution;
            run_tests(size, trials, rng_type, distribution, save_plots, plot_name, vm);
        }
    }

    return 0;
}
