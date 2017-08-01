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

#ifndef HIPRAND_KERNEL_NVCC_H_
#define HIPRAND_KERNEL_NVCC_H_

#ifndef QUALIFIERS
#define QUALIFIERS __forceinline__ __device__
#endif // QUALIFIERS

#include <hip/hip_runtime.h>
#include <type_traits>

#include "curand_kernel.h"

typedef curandState_t hiprandState_t;
typedef curandStateXORWOW_t hiprandStateXORWOW_t;
typedef curandStatePhilox4_32_10_t hiprandStatePhilox4_32_10_t;
typedef curandStateMRG32k3a_t hiprandStateMRG32k3a_t;
typedef curandStateMtgp32_t hiprandStateMtgp32_t;
typedef curandStateSobol32_t hiprandStateSobol32_t;
typedef curandStateScrambledSobol32_t hiprandStateScrambledSobol32_t;
typedef curandStateSobol64_t hiprandStateSobol64_t;
typedef curandStateScrambledSobol64_t hiprandStateScrambledSobol64_t;

typedef curandDiscreteDistribution_t hiprandDiscreteDistribution_t;

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
            hiprandStateSobol32_t,
            hiprandStateScrambledSobol32_t,
            hiprandStateSobol64_t,
            hiprandStateScrambledSobol64_t
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
    check_state_type<StateType>();
    curand_init(seed, subsequence, offset, state);
}

template<class StateType>
QUALIFIERS
void skipahead(unsigned long long n, StateType * state)
{
    // Defined in curand_kernel.h
    check_state_type<StateType>();
}

template<class StateType>
QUALIFIERS
void skipahead_sequence(unsigned long long n, StateType * state)
{
    // Defined in curand_kernel.h
    check_state_type<StateType>();
}

template<class StateType>
QUALIFIERS
unsigned int hiprand(StateType * state)
{
    check_state_type<StateType>();
    return curand(state);
}

QUALIFIERS
uint4 hiprand4(hiprandStatePhilox4_32_10_t * state)
{
    return curand4(state);
}

template<class StateType>
QUALIFIERS
float hiprand_uniform(StateType * state)
{
    check_state_type<StateType>();
    return curand_uniform(state);
}

QUALIFIERS
float4 hiprand_uniform4(hiprandStatePhilox4_32_10_t * state)
{
    return curand_uniform4(state);
}

template<class StateType>
QUALIFIERS
double hiprand_uniform_double(StateType * state)
{
    check_state_type<StateType>();
    return curand_uniform_double(state);
}

QUALIFIERS
double2 hiprand_uniform2_double(hiprandStatePhilox4_32_10_t * state)
{
    return curand_uniform2_double(state);
}

QUALIFIERS
double4 hiprand_uniform4_double(hiprandStatePhilox4_32_10_t * state)
{
    return curand_uniform4_double(state);
}

template<class StateType>
QUALIFIERS
float hiprand_normal(StateType * state)
{
    check_state_type<StateType>();
    return curand_normal(state);
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
    return curand_normal2(state);
}

QUALIFIERS
float4 hiprand_normal4(hiprandStatePhilox4_32_10_t * state)
{
    return curand_normal4(state);
}

template<class StateType>
QUALIFIERS
double hiprand_normal_double(StateType * state)
{
    check_state_type<StateType>();
    return curand_normal_double(state);
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
    return curand_normal2_double(state);
}

QUALIFIERS
double4 hiprand_normal4_double(hiprandStatePhilox4_32_10_t * state)
{
    return curand_normal4_double(state);
}

template<class StateType>
QUALIFIERS
float hiprand_log_normal(StateType * state,
                         float mean, float stddev)
{
    check_state_type<StateType>();
    return curand_log_normal(state, mean, stddev);
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
    return curand_log_normal2(state, mean, stddev);
}

QUALIFIERS
float4 hiprand_log_normal4(hiprandStatePhilox4_32_10_t * state,
                           float mean, float stddev)
{
    return curand_log_normal4(state, mean, stddev);
}

template<class StateType>
QUALIFIERS
double hiprand_log_normal_double(StateType * state,
                                 double mean, double stddev)
{
    check_state_type<StateType>();
    return curand_log_normal_double(state, mean, stddev);
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
    return curand_log_normal2_double(state, mean, stddev);
}

QUALIFIERS
double4 hiprand_log_normal4_double(hiprandStatePhilox4_32_10_t * state,
                                   double mean, double stddev)
{
    return curand_log_normal4_double(state, mean, stddev);
}

template<class StateType>
QUALIFIERS
uint hiprand_poisson(StateType * state, double lambda)
{
    check_state_type<StateType>();
    return curand_poisson(state, lambda);
}

QUALIFIERS
uint4 hiprand_poisson4(hiprandStatePhilox4_32_10_t * state, double lambda)
{
    return curand_poisson4(state, lambda);
}

template<class StateType>
QUALIFIERS
uint hiprand_discrete(StateType * state, hiprandDiscreteDistribution_t discrete_distribution)
{
    check_state_type<StateType>();
    return curand_discrete(state, discrete_distribution);
}

QUALIFIERS
uint4 hiprand_discrete4(hiprandStatePhilox4_32_10_t * state, hiprandDiscreteDistribution_t discrete_distribution)
{
    return curand_discrete4(state, discrete_distribution);
}

#endif // HIPRAND_KERNEL_H_
