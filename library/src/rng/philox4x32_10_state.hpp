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

#ifndef ROCRAND_RNG_PHILOX4X32_10_STATE_H_
#define ROCRAND_RNG_PHILOX4X32_10_STATE_H_

struct rocrand_philox4_32_10_state
{
    uint4 counter;
    uint2 key;

    __host__ __device__
    rocrand_philox4_32_10_state()
        : counter({0, 0, 0, 0}), key({0, 0})
    {

    }

    __host__ __device__
    rocrand_philox4_32_10_state(unsigned long long seed)
        : counter({0, 0, 0, 0})
    {
       key.x = static_cast<unsigned int>(seed);
       key.y = static_cast<unsigned int>(seed >> 32);
    }

    inline __host__ __device__
    void discard(unsigned int n)
    {
        uint4 temp = counter;
        counter.x += n;
        counter.y += (counter.x < temp.x ? 1 : 0);
        counter.z += (counter.y < temp.y ? 1 : 0);
        counter.w += (counter.z < temp.z ? 1 : 0);
    }

    inline __host__ __device__
    void discard()
    {
        if(++counter.x != 0) return;
        if(++counter.y != 0) return;
        if(++counter.z != 0) return;
        counter.w++;
    }

    inline __host__
    void reset()
    {
        counter = {0, 0, 0, 0};
    }

    inline __host__
    void set_seed(unsigned long long seed)
    {
        key.x = static_cast<unsigned int>(seed);
        key.y = static_cast<unsigned int>(seed >> 32);
        reset();
    }
};

#endif // ROCRAND_RNG_PHILOX4X32_10_STATE_H_
