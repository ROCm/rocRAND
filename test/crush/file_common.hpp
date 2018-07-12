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

#ifndef ROCRAND_FILE_COMMON_H_
#define ROCRAND_FILE_COMMON_H_

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <limits>

template<class RealType>
void rocrand_file_write_results(std::string file_name, RealType * data, size_t n)
{
    std::ofstream fout(file_name, std::ios_base::out | std::ios_base::trunc);
    if(fout.is_open())
    {
        fout << std::setprecision(std::numeric_limits<long double>::digits10 + 1);
        for(size_t i = 0; i < n; i++)
        {
            if (i == n - 1)
                fout << data[i];
            else
                fout << data[i] << " ";
        }
        std::cout << "File was written successfully" << std::endl;
    }
    else
    {
        std::cout << "File could not be opened" << std::endl;
    }
}

#endif // ROCRAND_FILE_COMMON_H_
