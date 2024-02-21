// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

const int LFSR113_N = 4; // 4 values
const int LFSR113_M = 32; // 32-bit each

const int LFSR113_SIZE = LFSR113_M * LFSR113_N * LFSR113_N;

const int LFSR113_JUMP_MATRICES = 32;
const int LFSR113_JUMP_LOG2     = 2;

const int LFSR113_SEQUENCE_JUMP_LOG2 = 55;

static unsigned int jump_matrices[LFSR113_JUMP_MATRICES][LFSR113_SIZE];
static unsigned int sequence_jump_matrices[LFSR113_JUMP_MATRICES][LFSR113_SIZE];

// Define uint4 so we don't need <hip/hip_runtime>.
typedef unsigned int uint4 __attribute__((vector_size(16)));

void copy_mat(unsigned int* dst, const unsigned int* src)
{
    for(int i = 0; i < LFSR113_SIZE; i++)
    {
        dst[i] = src[i];
    }
}

void copy_vec(unsigned int* dst, const unsigned int* src)
{
    for(int i = 0; i < LFSR113_N; i++)
    {
        dst[i] = src[i];
    }
}

void mul_mat_vec_inplace(const unsigned int* m, unsigned int* v)
{
    unsigned int r[LFSR113_N] = {0};
    for(int i = 0; i < LFSR113_N; i++)
    {
        for(int j = 0; j < LFSR113_M; j++)
        {
            if(v[i] & (1U << j))
            {
                for(int k = 0; k < LFSR113_N; k++)
                {
                    r[k] ^= m[LFSR113_N * (i * LFSR113_M + j) + k];
                }
            }
        }
    }
    copy_vec(v, r);
}

void mul_mat_mat_inplace(unsigned int* a, const unsigned int* b)
{
    for(int i = 0; i < LFSR113_N * LFSR113_M; i++)
    {
        mul_mat_vec_inplace(b, a + i * LFSR113_N);
    }
}

void mat_pow(unsigned int* a, const unsigned int* b, const unsigned long long power)
{
    // Identity matrix
    for(int i = 0; i < LFSR113_N; i++)
    {
        for(int j = 0; j < LFSR113_M; j++)
        {
            for(int k = 0; k < LFSR113_N; k++)
            {
                a[(i * LFSR113_M + j) * LFSR113_N + k] = ((i == k) ? (1 << j) : 0);
            }
        }
    }

    // Exponentiation by squaring
    unsigned int y[LFSR113_SIZE];
    copy_mat(y, b);
    for(unsigned long long p = power; p > 0; p >>= 1)
    {
        if(p & 1)
        {
            mul_mat_mat_inplace(a, y);
        }

        // Square the matrix
        unsigned int t[LFSR113_SIZE];
        copy_mat(t, y);
        mul_mat_mat_inplace(y, t);
    }
}

struct rocrand_lfsr113_state
{
    uint4 z;
    uint4 subsequence;

    void discard()
    {
        unsigned int b;

        b    = (((z[0] << 6) ^ z[0]) >> 13);
        z[0] = (((z[0] & 4294967294U) << 18) ^ b);

        b    = (((z[1] << 2) ^ z[1]) >> 27);
        z[1] = (((z[1] & 4294967288U) << 2) ^ b);

        b    = (((z[2] << 13) ^ z[2]) >> 21);
        z[2] = (((z[2] & 4294967280U) << 7) ^ b);

        b    = (((z[3] << 3) ^ z[3]) >> 12);
        z[3] = (((z[3] & 4294967168U) << 13) ^ b);
    }
};

void generate_matrices()
{
    unsigned int one_step[LFSR113_SIZE];
    for(int i = 0; i < LFSR113_N; ++i)
    {
        for(int j = 0; j < LFSR113_M; ++j)
        {
            rocrand_lfsr113_state state;
            const unsigned int    b = 1U << j;

            for(int k = 0; k < LFSR113_M; ++k)
            {
                state.z[k] = (i == k ? b : 0);
            }
            state.subsequence = uint4{0, 0, 0, 0};

            state.discard();

            for(int k = 0; k < LFSR113_M; ++k)
            {
                one_step[(i * LFSR113_M + j) * LFSR113_N + k] = state.z[k];
            }
        }
    }

    // Matrices for jumps within the same sequence (offset). Only 2^0 (one_step) to 2^31 powers needed.
    {
        unsigned int a[LFSR113_SIZE];
        unsigned int b[LFSR113_SIZE];
        copy_mat(a, one_step);

        copy_mat(jump_matrices[0], a);
        for(int k = 1; k < LFSR113_JUMP_MATRICES; k++)
        {
            copy_mat(b, a);
            mat_pow(a, b, (1 << LFSR113_JUMP_LOG2));
            copy_mat(jump_matrices[k], a);
        }
    }

    // Matrices for jumps within sequences. From 2^0 (one_step) to 2^55 powers needed.
    {
        unsigned int a[LFSR113_SIZE];
        unsigned int b[LFSR113_SIZE];
        copy_mat(a, one_step);

        // For 55: A^(2^27)
        mat_pow(b, a, 1ULL << (LFSR113_SEQUENCE_JUMP_LOG2 / 2));
        // For 55: (A^(2^27))^(2^28) = A^(2^55)
        mat_pow(a, b, 1ULL << (LFSR113_SEQUENCE_JUMP_LOG2 - LFSR113_SEQUENCE_JUMP_LOG2 / 2));

        copy_mat(sequence_jump_matrices[0], a);
        for(int k = 1; k < LFSR113_JUMP_MATRICES; k++)
        {
            copy_mat(b, a);
            mat_pow(a, b, (1 << LFSR113_JUMP_LOG2));
            copy_mat(sequence_jump_matrices[k], a);
        }
    }
}

void write_matrices(std::ofstream& fout, const std::string name, unsigned int* a, bool is_device)
{
    fout << "static const " << (is_device ? "__device__ " : "") << "unsigned int " << name
         << "[LFSR113_JUMP_MATRICES][LFSR113_SIZE] = {" << std::endl;
    for(int k = 0; k < LFSR113_JUMP_MATRICES; k++)
    {
        fout << "    {" << std::endl;
        for(int i = 0; i < LFSR113_M; i++)
        {
            fout << "        ";
            for(int j = 0; j < LFSR113_N * LFSR113_N; j++)
            {
                fout << a[k * LFSR113_SIZE + i * LFSR113_N * LFSR113_N + j] << ", ";
            }
            fout << std::endl;
        }
        fout << "    }," << std::endl;
    }
    fout << "};" << std::endl;
    fout << std::endl;
}

int main(int argc, char const* argv[])
{
    if(argc != 2 || std::string(argv[1]) == "--help")
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./lfsr113_precomputed_generator "
                     "../../library/include/rocrand/rocrand_lfsr113_precomputed.h"
                  << std::endl;
        return -1;
    }
    generate_matrices();

    const std::string file_path(argv[1]);
    std::ofstream     fout(file_path, std::ios_base::out | std::ios_base::trunc);
    fout << R"(// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_LFSR113_PRECOMPUTED_H_
#define ROCRAND_LFSR113_PRECOMPUTED_H_

// Auto-generated file. Do not edit!
// Generated by tools/lfsr113_precomputed_generator

)";

    fout << "#define LFSR113_N " << LFSR113_N << std::endl;
    fout << "#define LFSR113_M " << LFSR113_M << std::endl;
    fout << "#define LFSR113_SIZE (LFSR113_M * LFSR113_N * LFSR113_N)" << std::endl;
    fout << "#define LFSR113_JUMP_MATRICES " << LFSR113_JUMP_MATRICES << std::endl;
    fout << "#define LFSR113_JUMP_LOG2 " << LFSR113_JUMP_LOG2 << std::endl;
    fout << std::endl;

    write_matrices(fout,
                   "d_lfsr113_jump_matrices",
                   static_cast<unsigned int*>(&jump_matrices[0][0]),
                   true);
    write_matrices(fout,
                   "h_lfsr113_jump_matrices",
                   static_cast<unsigned int*>(&jump_matrices[0][0]),
                   false);

    write_matrices(fout,
                   "d_lfsr113_sequence_jump_matrices",
                   static_cast<unsigned int*>(&sequence_jump_matrices[0][0]),
                   true);
    write_matrices(fout,
                   "h_lfsr113_sequence_jump_matrices",
                   static_cast<unsigned int*>(&sequence_jump_matrices[0][0]),
                   false);

    fout << R"(
#endif // ROCRAND_LFSR113_PRECOMPUTED_H_
)";

    return 0;
}
