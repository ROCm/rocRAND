// S. Joe and F. Y. Kuo, Remark on Algorithm 659: Implementing Sobol's quasirandom
// sequence generator, 2003
// http://doi.acm.org/10.1145/641876.641879

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

using namespace std;

struct sobol_set
{
    uint32_t d;
    uint32_t s;
    uint32_t a;
    uint32_t m[18];
};

bool read_sobol_set(struct sobol_set * inputs, int N, const std::string name)
{
    std::ifstream infile(name ,std::ios::in);
    if (!infile) {
        std::cout << "Input file containing direction numbers cannot be found!\n";
        return false;
    }
    char buffer[1000];
    infile.getline(buffer,1000,'\n');

    for (int32_t i = 0; i < N; i++) {
        infile >> inputs[i].d >> inputs[i].s >> inputs[i].a;
        for (uint32_t j = 0; j < inputs[i].s; j++)
            infile >> inputs[i].m[j];
    }

    return true;
}

template<typename DirectionVectorType>
void init_direction_vectors(struct sobol_set * inputs, DirectionVectorType* directions, int n_directions, int n)
{
    constexpr uint32_t shift_adjust = (sizeof(DirectionVectorType) * 8) - 1;
    for (int j = 0 ; j < n_directions ; j++)
    {
        directions[j] = 1lu << (shift_adjust - j);
    }
    directions += n_directions;
    for (int i = 1; i < n ; i++) {
        int ix = i - 1;
        int s = inputs[ix].s;
        for (int j = 0; j < s ; j++)
        {
            directions[j] = (DirectionVectorType)inputs[ix].m[j] << (shift_adjust - j);
        }
        for (int j = s; j < n_directions ; j++)
        {
            directions[j] = directions[j - s] ^ (directions[j - s] >> s);
            for (int k = 1; k < s ; k++)
            {
                directions[j] ^= ((((DirectionVectorType)inputs[ix].a >> (s - 1 - k)) & 1) * directions[j - k]);
            }
        }
        directions += n_directions;
    }
}

template<typename DirectionVectorType>
void write_matrices(std::ofstream& fout, const std::string name, DirectionVectorType* directions, int32_t n, int32_t bits, bool is_device)
{
    fout << "static const ";
    fout << (is_device ? "__device__ " : "");
    fout << ((sizeof(DirectionVectorType) == 4) ? "unsigned int " : "unsigned long long int");
    fout << name << ((sizeof(DirectionVectorType) == 4) ? "[SOBOL32_N] = " : "[SOBOL64_N] = ");
    fout << std::endl;
    fout << "    {";
    fout << std::endl;
    fout << "        ";;
    for (int k = 0; k < n; k++)
    {
        fout << "0x";
        fout << std::hex << std::setw(sizeof(DirectionVectorType) * 2) << std::setfill('0') << directions[k] << ", ";
        if ((k + 1) % bits == 0 && k != 1)
            fout  << std::endl << "        ";
    }
    fout << std::endl;
    fout << "    };" << std::endl;
    fout << std::endl;
}

int main(int argc, char const *argv[])
{
    if (argc != 3 || std::string(argv[1]) == "--help")
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./sobol32_direction_vector_generator new-joe-kuo-6.21201 ../../library/include/rocrand_sobol32_precomputed.h" << std::endl;
        std::cout << "  (the source file can be downloaded here: http://web.maths.unsw.edu.au/~fkuo/sobol/)" << std::endl;
        return -1;
    }

    const std::string vector_file(argv[1]);
    uint32_t SOBOL_DIM = 20000;
    uint32_t SOBOL32_N = SOBOL_DIM * 32;
    struct sobol_set * inputs = new struct sobol_set[SOBOL_DIM];
    uint32_t * directions_32 = new uint32_t[SOBOL32_N];
    bool read = read_sobol_set(inputs, SOBOL_DIM, vector_file);

    if (read)
    {
        init_direction_vectors<uint32_t>(inputs, directions_32, 32, SOBOL_DIM);

        const std::string file_path(argv[2]);
        std::ofstream fout(file_path, std::ios_base::out | std::ios_base::trunc);
        fout << R"(// Copyright (c) 2017-2021 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_SOBOL32_PRECOMPUTED_H_
#define ROCRAND_SOBOL32_PRECOMPUTED_H_

// Auto-generated file. Do not edit!
// Generated by tools/sobol_direction_vector_generator

)";

        fout << "#define SOBOL_DIM " << SOBOL_DIM << std::endl;
        fout << "#define SOBOL32_N " << SOBOL32_N << std::endl;
        fout << std::endl;

        write_matrices(fout, "h_sobol32_direction_vectors", directions_32, SOBOL32_N, 32, false);

        fout << R"(
#endif // ROCRAND_SOBOL32_PRECOMPUTED_H_
)";
    }

    delete[] inputs;
    delete[] directions_32;

    return 0;
}
