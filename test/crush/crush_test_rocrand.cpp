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
#include <vector>
#include <string>
#include <chrono>
#include <array>
#include <iterator>
#include <random>
#include <algorithm>
#include <cstring>

#include <hip/hip_runtime.h>
#include <rocrand.h>

#include "file_common.hpp"
#include "cmdparser.hpp"

extern "C" {
#include "bbattery.h"
}

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
const size_t DEFAULT_RAND_N = 1024 * 1024 * 64;
#endif

void run_crush_test(const size_t size, const rocrand_rng_type rng_type)
{
    std::string filename = "rocrand_generate.txt";
    char * file = new char[filename.length() + 1];
    std::strcpy(file, filename.c_str());

    unsigned int * data;
    unsigned int * h_data = new unsigned int[size];
    double * h_data_double = new double[size];
    HIP_CHECK(hipMalloc((void **)&data, size * sizeof(unsigned int)));

    rocrand_generator generator;
    ROCRAND_CHECK(rocrand_create_generator(&generator, rng_type));
    // Make sure memory is allocated
    HIP_CHECK(hipDeviceSynchronize());

    ROCRAND_CHECK(rocrand_generate(generator, (unsigned int *) data, size));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(h_data, data, size * sizeof(unsigned int), hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());

    for(size_t i = 0; i < size; i++)
    {
        h_data_double[i] = static_cast<double>(h_data[i]) / (UINT_MAX + 1ULL);
    }
    delete[] h_data;

    rocrand_file_write_results(filename, h_data_double, size);

    rocrand_destroy_generator(generator);
    HIP_CHECK(hipFree(data));
    delete[] h_data_double;

    bbattery_SmallCrushFile(file);
    std::remove(file);
    delete[] file;
}

int main(int argc, char *argv[])
{
    cli::Parser parser(argc, argv);

    parser.set_optional<size_t>("size", "size", DEFAULT_RAND_N, "number of values");
    parser.set_optional<std::string>("engine", "engine", "philox", "random number engine");
    parser.run_and_exit_if_error();

    const size_t size = parser.get<size_t>("size");
    const std::string& engine = parser.get<std::string>("engine");

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

    if(engine == "philox")
    {
        std::cout << "philox4x32_10:" << std::endl;
        run_crush_test(size, ROCRAND_RNG_PSEUDO_PHILOX4_32_10);
        return 0;
    }
    else if(engine == "mrg32k3a")
    {
        std::cout << "mrg32k3a:" << std::endl;
        run_crush_test(size, ROCRAND_RNG_PSEUDO_MRG32K3A);
        return 0;
    }
    else if(engine == "xorwow")
    {
        std::cout << "xorwow:" << std::endl;
        run_crush_test(size, ROCRAND_RNG_PSEUDO_XORWOW);
        return 0;
    }
    else if(engine == "mtgp32")
    {
        std::cout << "mtgp32:" << std::endl;
        run_crush_test(size, ROCRAND_RNG_PSEUDO_MTGP32);
        return 0;
    }

    std::cerr << "Error: unknown random number engine '" << engine << "'" << std::endl;
    return -1;
}
