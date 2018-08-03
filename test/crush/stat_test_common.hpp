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

#ifndef STAT_TEST_COMMON_H_
#define STAT_TEST_COMMON_H_

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <numeric>
#include <utility>
#include <algorithm>

extern "C" {
#include "gofs.h"
#include "fdist.h"
#include "fbar.h"
#include "finv.h"
}

using distribution_func_type = std::function<double(double)>;

template<typename T>
double get_mean(const T * values, const size_t size)
{
    double mean = 0.0f;
    for (size_t i = 0; i < size; i++)
    {
        mean += static_cast<double>(values[i]);
    }
    return mean / size;
}

template<typename T>
double get_stddev(const T * values, const size_t size, double mean)
{
    double variance = 0.0f;
    for (size_t i = 0; i < size; i++)
    {
        const double x = static_cast<double>(values[i]) - mean;
        variance += x * x;
    }
    return std::sqrt(variance / size);
}

template<typename T>
void save_points_plots(const size_t size,
                       const size_t level1_tests,
                       const T * data,
                       const std::string plot_name)
{
    for (size_t level1_test = 0; level1_test < level1_tests; level1_test++)
    {
        std::ofstream fout;
        fout.open(plot_name + "-" + std::to_string(level1_test) + ".plot",
            std::ios_base::out | std::ios_base::trunc);

        fout << "set size square" << std::endl;

        if (std::is_integral<T>::value)
        {
            fout << "plot '-' with points pointtype 7 pointsize 0.15 notitle" << std::endl;

            const size_t x_offset = level1_test * size;
            const size_t y_offset = ((level1_test + 1 + level1_tests) % level1_tests) * size;
            for (size_t si = 0; si < size; si++)
            {
                const double r = 0.25;
                const double a = 2.0 * M_PI * si / size;
                const double x = data[x_offset + si] + r * std::cos(a);
                const double y = data[y_offset + si] + r * std::sin(a);
                fout << x << '\t' << y << std::endl;
            }
            fout << "e" << std::endl;
        }
        else
        {
            fout << "plot '-' with points pointtype 7 pointsize 0.15 notitle" << std::endl;

            const size_t x_offset = level1_test * size;
            const size_t y_offset = ((level1_test + 1 + level1_tests) % level1_tests) * size;
            for (size_t si = 0; si < size; si++)
            {
                const double x = data[x_offset + si];
                const double y = data[y_offset + si];
                fout << x << '\t' << y << std::endl;
            }
            fout << "e" << std::endl;
        }

        fout << "pause mouse close" << std::endl;
    }
}

template<typename T>
void analyze(const size_t size,
             const size_t level1_tests,
             const T * data,
             const bool save_plots,
             const std::string plot_name,
             const double mean, const double stddev,
             const distribution_func_type& distribution_func)__attribute__((cpu))
{
    if (save_plots)
    {
        save_points_plots(size, level1_tests, data, plot_name);
    }

    const double alpha = 0.05;

    double start = (mean - 6.0 * stddev);
    if (std::is_integral<T>::value)
    {
        // Use integral values for discrete distributions (e.g. Poisson)
        start = std::floor(start);
    }

    struct test_param
    {
        std::string name;
        double rejection_criterion;
        std::vector<double> ps;

        double cell_width;
        std::vector<double> nb_exp;
        std::vector<double> xs;
        std::vector<int> merged_count;
        std::vector<long> loc;
        long smin;
        long smax;
        long nb_classes;
    };

    const std::vector<size_t> max_cells_counts({ 1000, 100, 25 });
    const size_t tests = max_cells_counts.size();

    std::vector<test_param> ts(tests);

    for (size_t test = 0; test < tests; test++)
    {
        test_param& t = ts[test];

        const size_t cells_count = max_cells_counts[test];

        t.cell_width = 12.0 * stddev / cells_count;
        if (std::is_integral<T>::value)
        {
            // Use integral values for discrete distributions (e.g. Poisson)
            t.cell_width = std::ceil(t.cell_width);
        }

        t.nb_exp.resize(cells_count);
        t.xs.resize(cells_count);
        t.merged_count.resize(cells_count);
        t.loc.resize(cells_count);

        for (size_t ci = 0; ci < cells_count; ci++)
        {
            const double x0 = start + ci * t.cell_width;
            const double x1 = start + (ci + 1) * t.cell_width;
            const double expected = distribution_func(x1) - distribution_func(x0);
            t.nb_exp[ci] = expected * size;
            t.xs[ci] = x1;
            t.merged_count[ci] = 1;
        }

        t.smin = 0;
        t.smax = cells_count - 1;
        t.nb_classes = 0;

        // Merge classes (cells) with low probability to ensure that
        // the expected number of observation per class is at least 5
        gofs_MinExpected = 5.0;
        gofs_MergeClasses(t.nb_exp.data(), t.loc.data(), &t.smin, &t.smax, &t.nb_classes);

        for (long s = 0; s < static_cast<long>(cells_count); s++)
        {
            const long j = t.loc[s];
            if (j != s)
            {
                t.merged_count[j] += t.merged_count[s];
                t.merged_count[s] = 0;
            }
        }

        // When the chi-squared statistic exceeds the critical value (rejection_criterion),
        // we reject the null hypothesis ("observed random values follow a specific distribution")
        // with alpha significance level.
        t.rejection_criterion = finv_ChiSquare2(static_cast<long>(t.nb_classes - 1), 1.0 - alpha);
        t.name = "P" + std::to_string(t.nb_classes);

        t.ps.resize(level1_tests);
    }

    const int w = 12;
    const int w0 = 4;

    // Header
    {
        std::cout << "  ";
        std::cout << std::setw(w0) << "#";
        std::cout << std::setw(w) << "mean";
        std::cout << std::setw(w) << "stddev";

        for (size_t test = 0; test < tests; test++)
        {
            const test_param& t = ts[test];
            std::cout << std::setw(w) << t.name;
            std::cout << " ";
            std::cout << std::setw(w) << "p";
            std::cout << " ";
        }
        std::cout << std::endl;
        std::cout << "  ";
        std::cout << std::setw(w0) << "";
        std::cout << std::setw(w) << std::fixed << std::setprecision(3) << mean;
        std::cout << std::setw(w) << std::fixed << std::setprecision(3) << stddev;
        for (size_t test = 0; test < tests; test++)
        {
            const test_param& t = ts[test];
            std::cout << std::setw(w) << ("< " + std::to_string(static_cast<int>(t.rejection_criterion)));
            std::cout << " ";
            std::cout << std::setw(w) << "";
            std::cout << " ";
        }
        std::cout << std::endl << std::endl;
    }

    for (size_t level1_test = 0; level1_test < level1_tests; level1_test++)
    {
        std::cout << "  ";
        std::cout << std::setw(w0) << level1_test;
        const double test_mean = get_mean(&data[level1_test * size], size);
        const double test_stddev = get_stddev(&data[level1_test * size], size, test_mean);
        std::cout << std::setw(w) << std::fixed << std::setprecision(3) << test_mean;
        std::cout << std::setw(w) << std::fixed << std::setprecision(3) << test_stddev;

        for (size_t test = 0; test < tests; test++)
        {
            test_param& t = ts[test];

            const size_t cells_count = max_cells_counts[test];
            std::vector<long> count(cells_count, 0);

            for (size_t si = 0; si < size; si++)
            {
                const double v = data[level1_test * size + si];
                const long cell = static_cast<long>((v - start) / t.cell_width);
                if (cell >= 0 && cell < static_cast<long>(cells_count))
                {
                    count[cell]++;
                }
            }
            for (long s = 0; s < static_cast<long>(cells_count); s++)
            {
                const long j = t.loc[s];
                if (j != s)
                {
                    count[j] += count[s];
                    count[s] = 0;
                }
            }

            const double chi_squared = gofs_Chi2(const_cast<double *>(t.nb_exp.data()), count.data(), t.smin, t.smax);
            const double p = 1.0 - fdist_ChiSquare2(static_cast<long>(t.nb_classes - 1), 15, chi_squared);
            t.ps[level1_test] = p;

            std::cout << std::setw(w) << std::fixed << std::setprecision(3) << chi_squared;
            std::cout << (chi_squared < t.rejection_criterion ? " " : "*");
            std::cout << std::setw(w) << std::fixed << std::setprecision(3) << p;
            std::cout << (alpha < p ? " " : "*");

            if (save_plots)
            {
                std::ofstream fout;
                fout.open(plot_name + "-" + std::to_string(level1_test) + "-" + t.name + ".plot",
                    std::ios_base::out | std::ios_base::trunc);

                fout << "set arrow from " << mean << ", graph 0 to " << mean << ", graph 1 nohead lt 0 lc rgb 'blue'" << std::endl;
                fout << "set arrow from " << test_mean << ", graph 0 to " << test_mean << ", graph 1 nohead lt 0 lc rgb 'red'" << std::endl;
                fout << "plot '-' title 'observed' with fsteps, '-' title 'expected' with fsteps" << std::endl;

                for (long s = t.smin; s <= t.smax; s++)
                {
                    if (t.nb_exp[s] > 0.0)
                    {
                        const double v = count[s] / static_cast<double>(size) / t.merged_count[s];
                        if (s == t.smin)
                            fout << start << '\t' << v << std::endl;
                        fout << t.xs[s] << '\t' << v << std::endl;
                        if (s == t.smax)
                            fout << (start + cells_count * t.cell_width) << '\t' << v << std::endl;
                    }
                }
                fout << "e" << std::endl;

                for (long s = t.smin; s <= t.smax; s++)
                {
                    if (t.nb_exp[s] > 0.0)
                    {
                        const double v = t.nb_exp[s] / static_cast<double>(size) / t.merged_count[s];
                        if (s == t.smin)
                            fout << start << '\t' << v << std::endl;
                        fout << t.xs[s] << '\t' << v << std::endl;
                        if (s == t.smax)
                            fout << (start + cells_count * t.cell_width) << '\t' << v << std::endl;
                    }
                }
                fout << "e" << std::endl;

                fout << "pause mouse close" << std::endl;
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    {
        std::cout << "  ";
        std::cout << std::setw(w0) << "AD";
        std::cout << std::setw(w) << "";
        std::cout << std::setw(w) << "";
        for (size_t test = 0; test < tests; test++)
        {
            const test_param& t = ts[test];
            std::vector<double> ps(t.ps.begin(), t.ps.end());

            // Anderson-Darling test needs ordered values
            std::sort(ps.begin(), ps.end());

            const int n = level1_tests;
            const double a = gofs_AndersonDarling(ps.data() - 1, n);
            const double p = 1.0 - fdist_AndersonDarling2(n, a);

            std::cout << std::setw(w) << std::fixed << std::setprecision(3) << a;
            std::cout << " ";
            std::cout << std::setw(w) << std::fixed << std::setprecision(3) << p;
            std::cout << (alpha < p && p < (1.0 - alpha) ? " " : "*");
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

#endif // STAT_TEST_COMMON_H_
