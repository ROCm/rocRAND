#include <stdio.h>
#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <rocrand/rocrand.h>

#include "test_common.hpp"
#include "test_rocrand_common.hpp"
#include "test_utils_hipgraphs.hpp"

class rocrand_hipgraph_generate_tests : public ::testing::TestWithParam<rocrand_rng_type> {};

void test_float(std::function<rocrand_status(rocrand_generator, float*, size_t, float, float)> generate_fn, rocrand_rng_type rng_type)
{
    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator(
            &generator,
            rng_type
        )
    );

    ROCRAND_CHECK(rocrand_initialize_generator(generator));

    const size_t size = 12563;
    float mean = 5.0f;
    float stddev = 2.0f;
    float * data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    // Default stream does not support hipGraph stream capture, so create a non-blocking one
    hipStream_t stream = 0;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    rocrand_set_stream(generator, stream);

    hipGraphExec_t graph_instance;
    hipGraph_t graph = test_utils::createGraphHelper(stream);

    // Any sizes
    ROCRAND_CHECK(
        generate_fn(generator, data, 1, mean, stddev)
    );

    graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    test_utils::resetGraphHelper(graph, graph_instance, stream);

    // Any alignment
    ROCRAND_CHECK(
        generate_fn(generator, data+1, 2, mean, stddev)
    );

    graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    test_utils::resetGraphHelper(graph, graph_instance, stream);

    ROCRAND_CHECK(
        generate_fn(generator, data, size, mean, stddev)
    );

    graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    test_utils::cleanupGraphHelper(graph, graph_instance);
    HIP_CHECK(hipStreamDestroy(stream));
}

TEST_P(rocrand_hipgraph_generate_tests, normal_float_test)
{
    auto generator_fcn = [](rocrand_generator generator, float* output_data, size_t n, float mean, float stddev)
    {
        return rocrand_generate_normal(generator, output_data, n, mean, stddev);
    };

    test_float(generator_fcn, GetParam());
}

TEST_P(rocrand_hipgraph_generate_tests, log_normal_float_test)
{
    auto generator_fcn = [](rocrand_generator generator, float* output_data, size_t n, float mean, float stddev)
    {
        return rocrand_generate_log_normal(generator, output_data, n, mean, stddev);
    };

    test_float(generator_fcn, GetParam());
}

TEST_P(rocrand_hipgraph_generate_tests, uniform_float_test)
{
    const rocrand_rng_type rng_type = GetParam();

    rocrand_generator generator;
    ROCRAND_CHECK(
        rocrand_create_generator(
            &generator,
            rng_type
        )
    );

    ROCRAND_CHECK(rocrand_initialize_generator(generator));

    const size_t size = 12563;
    float * data;
    HIP_CHECK(hipMallocHelper(&data, size * sizeof(float)));
    HIP_CHECK(hipDeviceSynchronize());

    // Default stream does not support hipGraph stream capture, so create a non-blocking one
    hipStream_t stream = 0;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    rocrand_set_stream(generator, stream);
    
    hipGraphExec_t graph_instance;
    hipGraph_t graph = test_utils::createGraphHelper(stream);

    // Any sizes
    ROCRAND_CHECK(
        rocrand_generate_uniform(generator, data, 1)
    );

    graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    test_utils::resetGraphHelper(graph, graph_instance, stream);

    // Any alignment
    ROCRAND_CHECK(
        rocrand_generate_uniform(generator, data+1, 2)
    );

    graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);
    test_utils::resetGraphHelper(graph, graph_instance, stream);

    ROCRAND_CHECK(
        rocrand_generate_uniform(generator, data, size)
    );

    graph_instance = test_utils::endCaptureGraphHelper(graph, stream, true, true);

    HIP_CHECK(hipFree(data));
    ROCRAND_CHECK(rocrand_destroy_generator(generator));
    test_utils::cleanupGraphHelper(graph, graph_instance);
    HIP_CHECK(hipStreamDestroy(stream));
}

INSTANTIATE_TEST_SUITE_P(rocrand_hipgraph_generate_tests,
                        rocrand_hipgraph_generate_tests,
                        ::testing::ValuesIn(rng_types));
