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

/// \file cpp_utils.hpp
/// Utility templates to provide extended functionality to standard library types.

#ifndef ROCRAND_RNG_CPP_UTILS_HPP_
#define ROCRAND_RNG_CPP_UTILS_HPP_

#include <array>
#include <cstddef>
#include <limits>
#include <tuple>
#include <utility>

namespace cpp_utils
{

/// @brief Invoke a function on each element of a \c std::tuple separately.
/// @tparam Functor Type of the function.
/// @tparam Tuple Type of the \c std::tuple.
/// @param fun Functor to invoke.
/// @param tuple Tuple whose every element is passed to \c fun.
template<class Functor, class Tuple>
constexpr void visit_tuple(Functor&& fun, Tuple&& tuple)
{
    std::apply([&fun](auto&&... args) { ((fun(args)), ...); }, tuple);
}

template<class T>
struct array_size
{
    // always false
    static_assert(sizeof(T) == 0, "T must be an std::array");
};

template<class T, std::size_t N>
struct array_size<std::array<T, N>>
{
    static constexpr inline std::size_t value = N;
};

/// @brief Returns the element count (size) of an \ref std::array.
/// @tparam T Type of the \ref std::array.
template<class T>
constexpr std::size_t array_size_v = array_size<T>::value;

/// @brief Generates all combinations of values of the input arrays.
/// @tparam T Element type of the input arrays.
/// @tparam ...Ns Sizes of the input arrays.
/// @param ...inputs Input array args.
/// @return An array of arrays. Each array in the returned array contains
/// one unique combination of the values from the input arrays.
template<class T, std::size_t... Ns>
constexpr auto numeric_combinations(const std::array<T, Ns>... inputs)
{
    constexpr std::size_t dim = sizeof...(Ns);
    static_assert(dim >= 1, "There must be at least one argument.");
    constexpr std::size_t                   product = (Ns * ...);
    std::array<std::array<T, dim>, product> ret{};

    // keeps track of the column being written
    std::size_t col = 0;
    // keeps track of how many rows we have to write the same value in the current column
    // to generate all combinations.
    std::size_t stride = 1;
    // Invoke a lambda on all inputs, building the output column-by-column
    ((
         [&]
         {
             // inputs.size() should also work, but it doesn't with NVCC
             constexpr std::size_t N = array_size_v<std::decay_t<decltype(inputs)>>;
             static_assert(N > 0, "Each array must have at least one element.");
             for(std::size_t i = 0; i < product; ++i)
             {
                 ret[i][col] = inputs[i / stride % N];
             }
             stride *= N;
             ++col;
         }()),
     ...);

    return ret;
}

} // end namespace cpp_utils

#endif // ROCRAND_RNG_CPP_UTILS_HPP_
