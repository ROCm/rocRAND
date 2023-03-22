// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rng/cpp_utils.hpp"
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

TEST(rocrand_cpp_utils_tests, visit_tuple)
{
    std::tuple               t{1, std::size_t(2), (unsigned short)(3)};
    std::vector<std::string> results;

    cpp_utils::visit_tuple([&](auto&& val) { results.push_back(std::to_string(val)); }, t);

    ASSERT_EQ("1", results.at(0));
    ASSERT_EQ("2", results.at(1));
    ASSERT_EQ("3", results.at(2));
}

TEST(rocrand_cpp_utils_tests, tuple_type_index)
{
    using cpp_utils::tuple_type_index_v;

    static_assert(0 == tuple_type_index_v<int, std::tuple<int>>);
    static_assert(0 == tuple_type_index_v<int, std::tuple<int, int>>);
    static_assert(1 == tuple_type_index_v<int, std::tuple<float, int>>);
    static_assert(1 == tuple_type_index_v<int, std::tuple<float, int, int>>);
    static_assert(std::numeric_limits<std::size_t>::max()
                  == tuple_type_index_v<double, std::tuple<float, int, int>>);
    static_assert(std::numeric_limits<std::size_t>::max()
                  == tuple_type_index_v<double, std::tuple<>>);
}

TEST(rocrand_cpp_utils_tests, numeric_combinations)
{
    constexpr std::array A{1, 2, 3, 4};
    constexpr std::array B{10, 20, 30};
    constexpr std::array C{100, 200};

    constexpr auto combinations = cpp_utils::numeric_combinations(A, B, C);

    ASSERT_EQ(combinations.size(), A.size() * B.size() * C.size());

    const std::set combination_set(combinations.begin(), combinations.end());
    ASSERT_EQ(combinations.size(), combination_set.size()) << "Not all items are unique";

    for(auto [a, b, c] : combinations)
    {
        ASSERT_NE(std::find(A.begin(), A.end(), a), A.end()) << "Element not found in A";
        ASSERT_NE(std::find(B.begin(), B.end(), b), B.end()) << "Element not found in B";
        ASSERT_NE(std::find(C.begin(), C.end(), c), C.end()) << "Element not found in C";
    }
}
