#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>

#define HIP_CHECK(x) do { if((x)!=hipSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define ROCRAND_CHECK(x) do { if((x)!=ROCRAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

#if 1

void generate_lfsr113(const size_t size, unsigned int * output) {
    const rocrand_rng_type rng_type = ROCRAND_RNG_PSEUDO_LFSR113;

    rocrand_generator g = NULL;
    ROCRAND_CHECK(rocrand_create_generator(&g, rng_type));
    
    unsigned int * d_output;

    HIP_CHECK(hipMalloc(&d_output, sizeof(d_output[0]) * size));

    ROCRAND_CHECK(rocrand_generate(g, d_output, size));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output, d_output, sizeof(output[0]) * size, hipMemcpyDeviceToHost));

    ROCRAND_CHECK(rocrand_destroy_generator(g));
    HIP_CHECK(hipFree(d_output));
}

int main(int argc, char ** argv) {
    size_t size = 256;

    if (argc > 1)
        size = atoi(argv[1]);

    std::vector<unsigned int> output(size);

    generate_lfsr113(size, output.data());

    std::cout << "x,y" << std::endl;

    for (size_t i = 0; i < size; i+=2) {
        std::cout << output[i] << "," << output[i+1] << std::endl;
    }
}

#endif