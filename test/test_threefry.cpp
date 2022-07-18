#include <iostream>
#include <vector>
#include <chrono>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand_kernel.h>

#define HIP_CHECK(x) do { if((x)!=hipSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)
#define ROCRAND_CHECK(x) do { if((x)!=ROCRAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

void generate_threefry(const size_t size, unsigned int * output) {
    const rocrand_rng_type rng_type = ROCRAND_RNG_PSEUDO_THREEFRY;

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

void generate_threefry_uniform(const size_t size, float * output) {
    const rocrand_rng_type rng_type = ROCRAND_RNG_PSEUDO_THREEFRY;

    rocrand_generator g = NULL;
    ROCRAND_CHECK(rocrand_create_generator(&g, rng_type));
    
    float * d_output;

    HIP_CHECK(hipMalloc(&d_output, sizeof(d_output[0]) * size));

    ROCRAND_CHECK(rocrand_generate_uniform(g, d_output, size));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(output, d_output, sizeof(output[0]) * size, hipMemcpyDeviceToHost));

    ROCRAND_CHECK(rocrand_destroy_generator(g));
    HIP_CHECK(hipFree(d_output));
}

int main(int argc, char ** argv) {
    size_t size = 16;

    if (argc > 1)
        size = atoi(argv[1]);

    std::vector<unsigned int> output(size);
    // std::vector<float> output(size);

    generate_threefry(size, output.data());

    // std::cout << "x" << std::endl;
    for (size_t i = 0; i < size; i++) {
        std::cout << output[i] << std::endl;
    }
}