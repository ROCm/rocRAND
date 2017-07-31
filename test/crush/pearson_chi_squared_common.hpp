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

#ifndef PEARSON_CHI_SQUARED_COMMON_H_
#define PEARSON_CHI_SQUARED_COMMON_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <utility>
#include <algorithm>

#include <boost/program_options.hpp>

extern "C" {
#include "gofs.h"
#include "fdist.h"
#include "fbar.h"
#include "finv.h"
}

using distribution_func_type = std::function<double(double)>;

template<typename T>
void analyze(const size_t size,
             const size_t trials,
             const T * data,
             const bool save_plots,
             const std::string plot_name,
             const double mean, const double stddev,
             const distribution_func_type& distribution_func)
{
    const std::vector<size_t> cells_counts({ 1000, 100, 25 });
    // When the chi-squared statistic exceeds the critical value (rejection_criteria),
    // we reject the null hypothesis ("observed random values are * distribution")
    // with 90% significance level.
    const double significance_level = 0.9;
    std::vector<double> rejection_criteria;
    for (size_t cells_count : cells_counts)
    {
        const double c = finv_ChiSquare2(static_cast<long>(cells_count - 1), significance_level);
        rejection_criteria.push_back(c);
    }

    const int w = 14;
    const int w0 = 4;

    // Header
    {
        std::cout << "  ";
        std::cout << std::setw(w0) << "#";
        for (size_t cells_count : cells_counts)
        {
            std::cout << std::setw(w) << ("P" + std::to_string(cells_count));
            std::cout << " ";
        }
        std::cout << std::endl;
        std::cout << "  ";
        std::cout << std::setw(w0) << "";
        for (double c : rejection_criteria)
        {
            std::cout << std::setw(w) << ("< " + std::to_string(static_cast<int>(c)));
            std::cout << " ";
        }
        std::cout << std::endl << std::endl;
    }

    for (size_t trial = 0; trial < trials; trial++)
    {
        std::cout << "  ";
        std::cout << std::setw(w0) << trial;
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

            for (size_t si = 0; si < size; si++)
            {
                const double v = data[trial * size + si];
                const int cell = static_cast<int>((v - start) / cell_width);
                if (cell >= 0 && cell < cells_count)
                {
                    historgram[cell]++;
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
                const double observed = historgram[ci] / static_cast<double>(size);
                double expected = distribution_func(x1) - distribution_func(x0);
                if (expected == 0.0)
                    expected = 1e-12;
                chi_squared += (observed - expected) * (observed - expected) / expected;

                if (save_plots)
                {
                    fout << observed << "\t" << expected << std::endl;
                }
            }
            chi_squared *= size;

            std::cout << std::setw(w) << std::fixed << std::setprecision(5) << chi_squared;
            std::cout << (chi_squared < rejection_criterion ? " " : "*");
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

#endif // PEARSON_CHI_SQUARED_COMMON_H_
