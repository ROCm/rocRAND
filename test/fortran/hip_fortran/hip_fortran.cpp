#include <stdint.h>
#include <stddef.h>
#include <iostream>
#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

hipError_t _hipMalloc(void** ptr, size_t size)
{
    return hipMalloc(ptr, size);
}

hipError_t _hipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind)
{
    return hipMemcpy(dst, src, sizeBytes, kind);
}

hipError_t _hipFree(void* ptr)
{
    return hipFree(ptr);
}
    
#ifdef __cplusplus
}
#endif

