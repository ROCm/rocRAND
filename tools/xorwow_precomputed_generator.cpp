#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>


const int XORWOW_N = 5;  // 5 values
const int XORWOW_M = 32; // 32-bit each

const int XORWOW_SIZE = XORWOW_M * XORWOW_N * XORWOW_N;

const int XORWOW_JUMP_MATRICES = 32;
const int XORWOW_JUMP_LOG2 = 2;

const int XORWOW_SEQUENCE_JUMP_LOG2 = 67;

static unsigned int jump_matrices[XORWOW_JUMP_MATRICES][XORWOW_SIZE];
static unsigned int sequence_jump_matrices[XORWOW_JUMP_MATRICES][XORWOW_SIZE];


void copy_mat(unsigned int * dst, const unsigned int * src)
{
    for (int i = 0; i < XORWOW_SIZE; i++)
    {
        dst[i] = src[i];
    }
}

void copy_vec(unsigned int * dst, const unsigned int * src)
{
    for (int i = 0; i < XORWOW_N; i++)
    {
        dst[i] = src[i];
    }
}

void mul_mat_vec_inplace(const unsigned int * m, unsigned int * v)
{
    unsigned int r[XORWOW_N] = { 0 };
    for (int i = 0; i < XORWOW_N; i++)
    {
        for (int j = 0; j < XORWOW_M; j++)
        {
            if (v[i] & (1 << j))
            {
                for (int k = 0; k < XORWOW_N; k++)
                {
                    r[k] ^= m[XORWOW_N * (i * XORWOW_M + j) + k];
                }
            }
        }
    }
    copy_vec(v, r);
}

void mul_mat_mat_inplace(unsigned int * a, const unsigned int * b)
{
    for (int i = 0; i < XORWOW_N * XORWOW_M; i++)
    {
        mul_mat_vec_inplace(b, a + i * XORWOW_N);
    }
}

void mat_pow(unsigned int * a, const unsigned int * b, const unsigned long long power)
{
    // Identity matrix
    for (int i = 0; i < XORWOW_N; i++)
    {
        for (int j = 0; j < XORWOW_M; j++)
        {
            for (int k = 0; k < XORWOW_N; k++)
            {
                a[(i * XORWOW_M + j) * XORWOW_N + k] = ((i == k) ? (1 << j) : 0);
            }
        }
    }

    // Exponentiation by squaring
    unsigned int y[XORWOW_SIZE];
    copy_mat(y, b);
    for (unsigned long long p = power; p > 0; p >>= 1)
    {
        if (p & 1)
        {
            mul_mat_mat_inplace(a, y);
        }

        // Square the matrix
        unsigned int t[XORWOW_SIZE];
        copy_mat(t, y);
        mul_mat_mat_inplace(y, t);
    }
}

struct rocrand_xorwow_state
{
    // Xorshift values (160 bits)
    unsigned int x[5];
    // Weyl sequence value
    unsigned int d;

    void discard()
    {
        const unsigned int t = x[0] ^ (x[0] >> 2);
        x[0] = x[1];
        x[1] = x[2];
        x[2] = x[3];
        x[3] = x[4];
        x[4] = (x[4] ^ (x[4] << 4)) ^ (t ^ (t << 1));

        d += 362437;
    }
};

void generate_matrices()
{
    unsigned int one_step[XORWOW_SIZE];

    for (int i = 0; i < XORWOW_N; i++)
    {
        for (int j = 0; j < XORWOW_M; j++)
        {
            rocrand_xorwow_state state;
            const unsigned int b = 1 << j;
            for (int k = 0; k < XORWOW_N; k++)
            {
                state.x[k] = (i == k ? b : 0);
            }
            state.d = 0;

            state.discard();

            for (int k = 0; k < XORWOW_N; k++)
            {
                one_step[(i * XORWOW_M + j) * XORWOW_N + k] = state.x[k];
            }
        }
    }

    {
        unsigned int a[XORWOW_SIZE];
        unsigned int b[XORWOW_SIZE];
        copy_mat(a, one_step);

        copy_mat(jump_matrices[0], a);
        for (int k = 1; k < XORWOW_JUMP_MATRICES; k++)
        {
            copy_mat(b, a);
            mat_pow(a, b, (1 << XORWOW_JUMP_LOG2));
            copy_mat(jump_matrices[k], a);
        }
    }

    {
        unsigned int a[XORWOW_SIZE];
        unsigned int b[XORWOW_SIZE];
        copy_mat(a, one_step);

        // For 67: A^(2^33)
        mat_pow(b, a, 1ULL << (XORWOW_SEQUENCE_JUMP_LOG2 / 2));
        // For 67: (A^(2^33))^(2^34) = A^(2^67)
        mat_pow(a, b, 1ULL << (XORWOW_SEQUENCE_JUMP_LOG2 - XORWOW_SEQUENCE_JUMP_LOG2 / 2));

        copy_mat(sequence_jump_matrices[0], a);
        for (int k = 1; k < XORWOW_JUMP_MATRICES; k++)
        {
            copy_mat(b, a);
            mat_pow(a, b, (1 << XORWOW_JUMP_LOG2));
            copy_mat(sequence_jump_matrices[k], a);
        }
    }
}

void write_matrices(std::ofstream& fout, const std::string name, unsigned int * a, bool is_device)
{
    fout << "static const " << (is_device ? "__device__ " : "") << "unsigned int " << name << "[XORWOW_JUMP_MATRICES][XORWOW_SIZE] = {" << std::endl;
    for (int k = 0; k < XORWOW_JUMP_MATRICES; k++)
    {
        fout << "    {" << std::endl;
        for (int i = 0; i < XORWOW_M; i++)
        {
            fout << "        ";
            for (int j = 0; j < XORWOW_N * XORWOW_N; j++)
            {
                fout << a[k * XORWOW_SIZE + i * XORWOW_N * XORWOW_N + j] << ", ";
            }
            fout << std::endl;
        }
        fout << "    }," << std::endl;
    }
    fout << "};" << std::endl;
    fout << std::endl;
}


int main(int argc, char const *argv[]) {
    if (argc != 2 || std::string(argv[1]) == "--help")
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./xorwow_precomputed_generator ../../library/include/rocrand_xorwow_precomputed.h" << std::endl;
        return -1;
    }

    generate_matrices();

    const std::string file_path(argv[1]);
    std::ofstream fout(file_path, std::ios_base::out | std::ios_base::trunc);
    fout << R"(// Copyright (c) 2017 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_XORWOW_PRECOMPUTED_H_
#define ROCRAND_XORWOW_PRECOMPUTED_H_

// Auto-generated file. Do not edit!
// Generated by tools/xorwow_precomputed_generator

)";

    fout << "#define XORWOW_N " << XORWOW_N << std::endl;
    fout << "#define XORWOW_M " << XORWOW_M << std::endl;
    fout << "#define XORWOW_SIZE (XORWOW_M * XORWOW_N * XORWOW_N)" << std::endl;
    fout << "#define XORWOW_JUMP_MATRICES " << XORWOW_JUMP_MATRICES << std::endl;
    fout << "#define XORWOW_JUMP_LOG2 " << XORWOW_JUMP_LOG2 << std::endl;
    fout << std::endl;

    write_matrices(fout, "d_xorwow_jump_matrices",
        static_cast<unsigned int *>(&jump_matrices[0][0]), true);
    write_matrices(fout, "h_xorwow_jump_matrices",
        static_cast<unsigned int *>(&jump_matrices[0][0]), false);

    write_matrices(fout, "d_xorwow_sequence_jump_matrices",
        static_cast<unsigned int *>(&sequence_jump_matrices[0][0]), true);
    write_matrices(fout, "h_xorwow_sequence_jump_matrices",
        static_cast<unsigned int *>(&sequence_jump_matrices[0][0]), false);

    fout << R"(
#endif // ROCRAND_XORWOW_PRECOMPUTED_H_
)";

    return 0;
}
