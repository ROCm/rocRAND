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

#ifndef HIPRAND_KERNEL_HCC_H_
#define HIPRAND_KERNEL_HCC_H_

#ifndef QUALIFIERS
#define QUALIFIERS __forceinline__ __device__
#endif // QUALIFIERS

#include <hip/hip_runtime.h>

#define FQUALIFIERS QUALIFIERS
#include "rocrand_kernel.h"

typedef rocrand_state_xorwow hiprandState_t;
typedef rocrand_state_xorwow hiprandStateXORWOW_t;
typedef rocrand_state_philox4x32_10 hiprandStatePhilox4_32_10_t;
typedef rocrand_state_mrg32k3a hiprandStateMRG32k3a_t;
typedef rocrand_state_mtgp32 hiprandStateMtgp32_t;
typedef rocrand_state_sobol32 hiprandStateSobol32_t;

typedef rocrand_discrete_distribution hiprandDiscreteDistribution_t;

typedef unsigned int hiprandDirectionVectors32_t[32];

typedef mtgp32_param mtgp32_kernel_params_t;

template<typename T, typename... R>
struct is_any_of : std::false_type { };

template<typename T, typename F>
struct is_any_of<T, F> : std::is_same<T, F> { };

template<typename T, typename F, typename... R>
struct is_any_of<T, F, R...>
    : std::integral_constant<
        bool,
        std::is_same<T, F>::value || is_any_of<T, R...>::value
      >
{ };

template<typename StateType>
QUALIFIERS
void check_state_type()
{
    static_assert(
        is_any_of<
            StateType,
            hiprandState_t,
            hiprandStateXORWOW_t,
            hiprandStatePhilox4_32_10_t,
            hiprandStateMRG32k3a_t,
            hiprandStateMtgp32_t,
            hiprandStateSobol32_t
        >::value,
        "StateType is not a hipRAND generator state"
    );
}

template<class StateType>
QUALIFIERS
void hiprand_init(const unsigned long long seed,
                  const unsigned long long subsequence,
                  const unsigned long long offset,
                  StateType * state)
{
    static_assert(
        !std::is_same<
            StateType,
            hiprandStateMtgp32_t
        >::value,
        "hiprandStateMtgp32_t does not have hiprand_init function, "
        "check hiprandMakeMTGP32KernelState() host function"
    );
    static_assert(
        !is_any_of<
            StateType,
            hiprandStateSobol32_t
        >::value,
        "Quasirandom generators use different hiprand_init() function"
    );
    check_state_type<StateType>();
    rocrand_init(seed, subsequence, offset, state);
}

QUALIFIERS
void hiprand_init(hiprandDirectionVectors32_t direction_vectors,
                  unsigned int offset,
                  hiprandStateSobol32_t * state)
{
    rocrand_init(direction_vectors, offset, state);
}

template<class StateType>
QUALIFIERS
void skipahead(unsigned long long n, StateType * state)
{
    static_assert(
        !std::is_same<
            StateType,
            hiprandStateMtgp32_t
        >::value,
        "hiprandStateMtgp32_t does not have skipahead function"
    );
    // Defined in rocrand_kernel.h
    check_state_type<StateType>();
}

template<class StateType>
QUALIFIERS
void skipahead_sequence(unsigned long long n, StateType * state)
{
    static_assert(
        !is_any_of<
            StateType,
            hiprandStateMtgp32_t,
            hiprandStateSobol32_t
        >::value,
        "StateType does not have skipahead_sequence function"
    );
    check_state_type<StateType>();
    skipahead_subsequence(n, state);
}

template<class StateType>
QUALIFIERS
void skipahead_subsequence(unsigned long long n, StateType * state)
{
    static_assert(
        is_any_of<
            StateType,
            hiprandStateMRG32k3a_t
        >::value,
        "Used StateType is not supported"
    );
    skipahead_subsequence(n, state);
}

template<class StateType>
QUALIFIERS
unsigned int hiprand(StateType * state)
{
    check_state_type<StateType>();
    return rocrand(state);
}

QUALIFIERS
uint4 hiprand4(hiprandStatePhilox4_32_10_t * state)
{
    return rocrand4(state);
}

template<class StateType>
QUALIFIERS
float hiprand_uniform(StateType * state)
{
    check_state_type<StateType>();
    return rocrand_uniform(state);
}

QUALIFIERS
float4 hiprand_uniform4(hiprandStatePhilox4_32_10_t * state)
{
    return rocrand_uniform4(state);
}

template<class StateType>
QUALIFIERS
double hiprand_uniform_double(StateType * state)
{
    check_state_type<StateType>();
    return rocrand_uniform_double(state);
}

QUALIFIERS
double2 hiprand_uniform2_double(hiprandStatePhilox4_32_10_t * state)
{
    return rocrand_uniform_double2(state);
}

QUALIFIERS
double4 hiprand_uniform4_double(hiprandStatePhilox4_32_10_t * state)
{
    return rocrand_uniform_double4(state);
}

template<class StateType>
QUALIFIERS
float hiprand_normal(StateType * state)
{
    check_state_type<StateType>();
    return rocrand_normal(state);
}

template<class StateType>
QUALIFIERS
float2 hiprand_normal2(StateType * state)
{
    check_state_type<StateType>();
    static_assert(
        is_any_of<
            StateType,
            hiprandStateXORWOW_t,
            hiprandStatePhilox4_32_10_t,
            hiprandStateMRG32k3a_t
        >::value,
        "Used StateType is not supported"
    );
    return rocrand_normal2(state);
}

QUALIFIERS
float4 hiprand_normal4(hiprandStatePhilox4_32_10_t * state)
{
    return rocrand_normal4(state);
}

template<class StateType>
QUALIFIERS
double hiprand_normal_double(StateType * state)
{
    check_state_type<StateType>();
    return rocrand_normal_double(state);
}

template<class StateType>
QUALIFIERS
double2 hiprand_normal2_double(StateType * state)
{
    check_state_type<StateType>();
    static_assert(
        is_any_of<
            StateType,
            hiprandStateXORWOW_t,
            hiprandStatePhilox4_32_10_t,
            hiprandStateMRG32k3a_t
        >::value,
        "Used StateType is not supported"
    );
    return rocrand_normal_double2(state);
}

QUALIFIERS
double4 hiprand_normal4_double(hiprandStatePhilox4_32_10_t * state)
{
    return rocrand_normal_double4(state);
}

template<class StateType>
QUALIFIERS
float hiprand_log_normal(StateType * state,
                         float mean, float stddev)
{
    check_state_type<StateType>();
    return rocrand_log_normal(state, mean, stddev);
}

template<class StateType>
QUALIFIERS
float2 hiprand_log_normal2(StateType * state,
                          float mean, float stddev)
{
    check_state_type<StateType>();
    static_assert(
        is_any_of<
            StateType,
            hiprandStateXORWOW_t,
            hiprandStatePhilox4_32_10_t,
            hiprandStateMRG32k3a_t
        >::value,
        "Used StateType is not supported"
    );
    return rocrand_log_normal2(state, mean, stddev);
}

QUALIFIERS
float4 hiprand_log_normal4(hiprandStatePhilox4_32_10_t * state,
                           float mean, float stddev)
{
    return rocrand_log_normal4(state, mean, stddev);
}

template<class StateType>
QUALIFIERS
double hiprand_log_normal_double(StateType * state,
                                 double mean, double stddev)
{
    check_state_type<StateType>();
    return rocrand_log_normal_double(state, mean, stddev);
}

template<class StateType>
QUALIFIERS
double2 hiprand_log_normal2_double(StateType * state,
                                   double mean, double stddev)
{
    check_state_type<StateType>();
    static_assert(
        is_any_of<
            StateType,
            hiprandStateXORWOW_t,
            hiprandStatePhilox4_32_10_t,
            hiprandStateMRG32k3a_t
        >::value,
        "Used StateType is not supported"
    );
    return rocrand_log_normal_double2(state, mean, stddev);
}

QUALIFIERS
double4 hiprand_log_normal4_double(hiprandStatePhilox4_32_10_t * state,
                                   double mean, double stddev)
{
    return rocrand_log_normal_double4(state, mean, stddev);
}

template<class StateType>
QUALIFIERS
uint hiprand_poisson(StateType * state, double lambda)
{
    check_state_type<StateType>();
    return rocrand_poisson(state, lambda);
}

QUALIFIERS
uint4 hiprand_poisson4(hiprandStatePhilox4_32_10_t * state, double lambda)
{
    return rocrand_poisson4(state, lambda);
}

template<class StateType>
QUALIFIERS
uint hiprand_discrete(StateType * state, hiprandDiscreteDistribution_t discrete_distribution)
{
    check_state_type<StateType>();
    return rocrand_discrete(state, discrete_distribution);
}

QUALIFIERS
uint4 hiprand_discrete4(hiprandStatePhilox4_32_10_t * state, hiprandDiscreteDistribution_t discrete_distribution)
{
    return rocrand_discrete4(state, discrete_distribution);
}

#endif // HIPRAND_KERNEL_HCC_H_
