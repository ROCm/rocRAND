#ifndef ROCRAND_ROCRANDAPI_H_
#define ROCRAND_ROCRANDAPI_H_

/// \cond ROCRAND_DOCS_MACRO
#ifndef ROCRANDAPI
    // device symbols are not marked with ROCRANDAPI (they are not exported)
    // but clang warns on host symbols if they are marked with dllexport/dllimport
    // during device compilation.
    #if defined(__HIP_DEVICE_COMPILE__)
        #define ROCRANDAPI
    #elif defined(_WIN32)
        #ifdef rocrand_EXPORTS
            /* We are building this library */
            #define ROCRANDAPI __declspec(dllexport)
        #else
            /* We are using this library */
            #define ROCRANDAPI __declspec(dllimport)
        #endif
    #else
        #define ROCRANDAPI __attribute__((visibility("default")))
    #endif
#endif
/// \endcond

#endif // ROCRAND_ROCRANDAPI_H_
