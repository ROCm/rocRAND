// Copyright (c) 2017-2023 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCRAND_RNG_GENERATOR_TYPES_H_
#define ROCRAND_RNG_GENERATOR_TYPES_H_

#include "generator_type.hpp"

#include "lfsr113.hpp"
#include "mrg31k3p.hpp"
#include "mrg32k3a.hpp"
#include "mt19937.hpp"
#include "mtgp32.hpp"
#include "philox4x32_10.hpp"
#include "sobol.hpp"
#include "threefry2x32_20.hpp"
#include "threefry2x64_20.hpp"
#include "threefry4x32_20.hpp"
#include "threefry4x64_20.hpp"
#include "xorwow.hpp"

extern template struct rocrand_generator_type<rocrand_lfsr113>;
extern template struct rocrand_generator_type<rocrand_mrg31k3p>;
extern template struct rocrand_generator_type<rocrand_mrg31k3p_host>;
extern template struct rocrand_generator_type<rocrand_mrg32k3a>;
extern template struct rocrand_generator_type<rocrand_mt19937>;
extern template struct rocrand_generator_type<rocrand_mtgp32>;
extern template struct rocrand_generator_type<rocrand_philox4x32_10>;
extern template struct rocrand_generator_type<rocrand_philox4x32_10_host>;
extern template struct rocrand_generator_type<rocrand_scrambled_sobol32>;
extern template struct rocrand_generator_type<rocrand_scrambled_sobol64>;
extern template struct rocrand_generator_type<rocrand_sobol32>;
extern template struct rocrand_generator_type<rocrand_sobol64>;
extern template struct rocrand_generator_type<rocrand_threefry2x32_20>;
extern template struct rocrand_generator_type<rocrand_threefry2x64_20>;
extern template struct rocrand_generator_type<rocrand_threefry4x32_20>;
extern template struct rocrand_generator_type<rocrand_threefry4x64_20>;
extern template struct rocrand_generator_type<rocrand_xorwow>;

#endif // ROCRAND_RNG_GENERATOR_TYPES_H_
