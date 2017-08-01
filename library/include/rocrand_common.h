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

/*
Copyright 2010-2011, D. E. Shaw Research.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions, and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of D. E. Shaw Research nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef ROCRAND_COMMON_H_
#define ROCRAND_COMMON_H_

#define ROCRAND_2POW32_INV (2.3283064e-10f)
#define ROCRAND_2POW32_INV_DOUBLE (2.3283064365386963e-10) 
#define ROCRAND_2POW32_INV_2PI (2.3283064e-10f * 6.2831855f)
#define ROCRAND_2POW53_INV_DOUBLE (1.1102230246251565e-16)
#define ROCRAND_PI_DOUBLE  (3.1415926535897932)
#define ROCRAND_2PI (6.2831855f)
#define ROCRAND_NORM_DOUBLE (2.3283065498378288e-10)
#define ROCRAND_SQRT2 (1.4142135f)
#define ROCRAND_SQRT2_DOUBLE (1.4142135623730951)

// nextafterf(float, float) is not implemented for device on HIP/HCC platform
#if defined(__HIP_PLATFORM_HCC__) && defined(__HIP_DEVICE_COMPILE__)
#ifndef ROCRAND_PLATFORM_HCC_NEXTAFTERF
#define ROCRAND_PLATFORM_HCC_NEXTAFTERF
inline __forceinline__ __device__
float nextafterf(const float from, const float to)
{
    if(from == to) return to;
    // (2.3283064e-10f) is float min value
    return fminf(from + (2.3283064e-10f), to);
}
#endif
#endif // defined(__HIP_PLATFORM_HCC__) && defined(__HIP_DEVICE_COMPILE__)

namespace rocrand_device {
namespace detail {

// This helps access fields of engine's internal state which
// saves floats and doubles generated using the Box–Muller transform
template<typename Engine>
struct engine_boxmuller_helper
{
    static FQUALIFIERS
    bool has_float(const Engine * engine)
    {
        return engine->m_state.boxmuller_float_state != 0;
    }

    static FQUALIFIERS
    float get_float(Engine * engine)
    {
        engine->m_state.boxmuller_float_state = 0;
        return engine->m_state.boxmuller_float;
    }

    static FQUALIFIERS
    void save_float(Engine * engine, float f)
    {
        engine->m_state.boxmuller_float_state = 1;
        engine->m_state.boxmuller_float = f;
    }

    static FQUALIFIERS
    bool has_double(const Engine * engine)
    {
        return engine->m_state.boxmuller_double_state != 0;
    }

    static FQUALIFIERS
    float get_double(Engine * engine)
    {
        engine->m_state.boxmuller_double_state = 0;
        return engine->m_state.boxmuller_double;
    }

    static FQUALIFIERS
    void save_double(Engine * engine, double d)
    {
        engine->m_state.boxmuller_double_state = 1;
        engine->m_state.boxmuller_double = d;
    }
};

} // end namespace detail
} // end namespace rocrand_device

#endif // ROCRAND_COMMON_H_
