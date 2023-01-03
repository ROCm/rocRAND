#ifndef ROCRAND_ROCRANDAPI_H_
#define ROCRAND_ROCRANDAPI_H_

/// \cond ROCRAND_DOCS_MACRO
#ifndef ROCRANDAPI
    #ifdef WIN32
        #ifdef rocrand_EXPORTS
            /* We are building this library */
            #define ROCRANDAPI __declspec(dllexport)
        #else
            /* We are using this library */
            #define ROCRANDAPI __declspec(dllimport)
        #endif
    #else
        #define ROCRANDAPI
    #endif
#endif
/// \endcond

#endif // ROCRAND_ROCRANDAPI_H_
