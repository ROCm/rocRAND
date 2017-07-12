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

#ifndef ROCRAND_LOG_NORMAL_H_
#define ROCRAND_LOG_NORMAL_H_

#ifndef FQUALIFIERS
#define FQUALIFIERS __forceinline__ __device__
#endif // FQUALIFIERS

#include "rocrand_philox4x32_10.h"
#include "rocrand_normal.h"

FQUALIFIERS
float rocrand_log_normal(rocrand_state_philox4x32_10 * state, float mean, float stddev)
{
    if(state->boxmuller_float_state != 0)
    {
        state->boxmuller_float_state = 0;
        return expf(mean + (stddev * state->boxmuller_float));
    }
    float2 r = detail::normal_distribution2(rocrand(state), rocrand(state));
    state->boxmuller_float_state = 1;
    state->boxmuller_float = r.y;
    return expf(mean + (stddev * r.x));
}

FQUALIFIERS
float2 rocrand_log_normal2(rocrand_state_philox4x32_10 * state, float mean, float stddev)
{
    float2 r = detail::normal_distribution2(rocrand(state), rocrand(state));
    return float2 {
        expf(mean + (stddev * r.x)),
        expf(mean + (stddev * r.y))
    };
}

FQUALIFIERS
float4 rocrand_log_normal4(rocrand_state_philox4x32_10 * state, float mean, float stddev)
{
    float4 r = detail::normal_distribution4(rocrand4(state));
    return float4 {
        expf(mean + (stddev * r.x)),
        expf(mean + (stddev * r.y)),
        expf(mean + (stddev * r.z)),
        expf(mean + (stddev * r.w))
    };
}

FQUALIFIERS
double rocrand_log_normal_double(rocrand_state_philox4x32_10 * state, double mean, double stddev)
{
    if(state->boxmuller_double_state != 0)
    {
        state->boxmuller_double_state = 0;
        return exp(mean + (stddev * state->boxmuller_double));
    }
    double2 r = detail::normal_distribution_double2(rocrand4(state));
    state->boxmuller_double_state = 1;
    state->boxmuller_double = r.y;
    return exp(mean + r.x * stddev);
}

FQUALIFIERS
double2 rocrand_log_normal_double2(rocrand_state_philox4x32_10 * state, double mean, double stddev)
{
    double2 r = detail::normal_distribution_double2(rocrand4(state));
    return double2 {
        exp(mean + (stddev * r.x)),
        exp(mean + (stddev * r.y))
    };
}

#endif // ROCRAND_LOG_NORMAL_H_
