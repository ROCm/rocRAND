#include <cstdio>
#include <fstream>
#include <ios>
#include <iostream>
#include <vector>
#include <string.h>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>

#define HIP_CHECK(x) do { if((x)!=hipSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define ROCRAND_CHECK(x) do { if((x)!=ROCRAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

void generate_data(rocrand_rng_type rng_type) {
    const size_t size = 1e11;
    const size_t nb_run = 50;

    size_t run_size = size / nb_run;

    rocrand_generator g = NULL;
    ROCRAND_CHECK(rocrand_create_generator(&g, rng_type));

    unsigned int * output;
    unsigned int * d_output;

    for (size_t i = 0; i < size; i += run_size) {
        output = (unsigned int *) malloc(sizeof(output[0]) * run_size);
        HIP_CHECK(hipMalloc(&d_output, sizeof(d_output[0]) * run_size));
    
        ROCRAND_CHECK(rocrand_generate(g, d_output, run_size));
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemcpy(output, d_output, sizeof(output[0]) * run_size, hipMemcpyDeviceToHost));

        fwrite(output, sizeof(unsigned int), run_size, stdout);

        free(output);
    }

    ROCRAND_CHECK(rocrand_destroy_generator(g));
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " --engine [engine_name]" << std::endl;
        std::cerr << "Available engines:" << std::endl;
        std::cerr << "\t- mrg32k3a" << std::endl;
        std::cerr << "\t- mtgp32" << std::endl;
        std::cerr << "\t- philox" << std::endl;
        std::cerr << "\t- threefry" << std::endl;
        std::cerr << "\t- xorwow" << std::endl;
        return 1;
    }

    if (strcmp(argv[1], "--engine") == 0) {
        if (strcmp(argv[2], "mrg32k3a") == 0) {
            generate_data(ROCRAND_RNG_PSEUDO_MRG32K3A);
        } else if (strcmp(argv[2], "mtgp32") == 0) {
            generate_data(ROCRAND_RNG_PSEUDO_MTGP32);
        } else if (strcmp(argv[2], "philox") == 0) {
            generate_data(ROCRAND_RNG_PSEUDO_PHILOX4_32_10);
        } else if (strcmp(argv[2], "threefry") == 0) {
            generate_data(ROCRAND_RNG_PSEUDO_THREEFRY);
        } else if (strcmp(argv[2], "xorwow") == 0) {
            generate_data(ROCRAND_RNG_PSEUDO_XORWOW);
        } else {
            std::cerr << "Error: Bad engine name." << std::endl;
            return 1;
        }
    }

    return 0;
}