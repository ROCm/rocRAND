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

template<class T, class Tuple>
class tuple_type_index
{
    template<std::size_t... Indices>
    static constexpr std::size_t get_type_index(std::index_sequence<Indices...>)
    {
        constexpr auto not_found_value = std::numeric_limits<std::size_t>::max();
        std::size_t    idx             = not_found_value;
        ((
             [&idx]
             {
                 if(std::is_same_v<T, std::tuple_element_t<Indices, Tuple>>
                    && idx == not_found_value)
                 {
                     idx = Indices;
                 }
             }()),
         ...);
        return idx;
    }

public:
    static inline constexpr std::size_t value
        = get_type_index(std::make_index_sequence<std::tuple_size_v<Tuple>>{});
};

/// @brief Gets the index of type \c T first occurring in the type list of a \c std::tuple.
/// If not present, returns \c std::numeric_limits<std::size_t>::max()
/// @tparam T Type whose index is searched.
/// @tparam Tuple Type of the \c std::tuple.
template<class T, class Tuple>
constexpr std::size_t tuple_type_index_v = tuple_type_index<T, Tuple>::value;

} // end namespace cpp_utils

#endif // ROCRAND_RNG_CPP_UTILS_HPP_
