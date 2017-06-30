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

#include <hip/hip_runtime.h>
#include "rng/base_generator.hpp"
#include <rocrand.h>
#include <new>

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */
    
rocrand_status ROCRANDAPI 
rocrand_create_generator(rocrand_generator *generator, rocrand_rng_type rng_type) 
{
    rocrand_status status = ROCRAND_STATUS_SUCCESS;
    
    try 
    {
        *generator = new rocrand_base_generator(rng_type);
    }
    catch(const std::bad_alloc& e)
    {
        return ROCRAND_STATUS_INTERNAL_ERROR;
    }
    catch(rocrand_status status)
    {
        return status;
    }

    return status;
}

rocrand_status ROCRANDAPI 
rocrand_destroy_generator(rocrand_generator generator) 
{
    rocrand_status status = ROCRAND_STATUS_SUCCESS;
    
    try 
    {
        delete(generator);
    }
    catch(rocrand_status status)
    {
        return status;
    }
    
    return status;
}
    
#if defined(__cplusplus)
}
#endif /* __cplusplus */
