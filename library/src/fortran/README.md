# rocRAND Fortran

This library provides a pure Fortran interface for the rocRAND/hipRAND API.

This interface is intended to target only Host API functions, and provides a 1:1 mapping to
the C Host API functions in rocRAND and hipRAND. For documentation of these functions, please
refer to the C Host API functions documentation.

## Build and Install

The Fortran interface is installed as part of the rocRAND package. Simply add the build
option `-DBUILD_FORTRAN_WRAPPER=ON` when configuring cmake, as below:

```
cmake -DBUILD_FORTRAN_WRAPPER=ON ../.
```

## Running Unit Tests

```
# Go to rocRAND build directory
cd rocRAND; cd build

# To run unit tests for Fortran interface
./test/test_rocrand_fortran_wrapper
./test/test_hiprand_fortran_wrapper
```

## Usage

Below is an example of writing a simple Fortran program that generates a set of uniform values.

```
integer(kind =8) :: gen
real, target, dimension(128) :: h_x
type(c_ptr) :: d_x
integer(c_int) :: status
integer(c_size_t), parameter :: output_size = 128
status = hipMalloc(d_x, output_size * sizeof(h_x(1)))
status = rocrand_create_generator(gen, ROCRAND_RNG_PSEUDO_DEFAULT)
status = rocrand_generate_uniform(gen, d_x, output_size)
status = hipMemcpy(c_loc(h_x), d_x, output_size * sizeof(h_x(1)), hipMemcpyDeviceToHost)
status = hipFree(d_x)
status = rocrand_destroy_generator(gen)
```

And when compiling the source code with a Fortran compiler, the following should be linked.
`gfortran` will be used as an example below, however other Fortran compilers should work.

For rocRAND Fortran interface:
```
gfortran <input-file>.f90 hip_m.f90 rocrand_m.f90  -lrocrand_fortran -lrocrand
# If compiling on a NVCC platform, link CUDA libraries (-lcuda -lcudart)
# If compiling on an AMD platform, link HIP library (-L${HIP_ROOT_DIR}/lib -lhip_hcc)
```

For hipRAND Fortran interface:
```
gfortran <input-file>.f90 hip_m.f90 hiprand_m.f90  -lhiprand_fortran -lhiprand
# If compiling on a NVCC platform, link CUDA and CURAND libraries (-lcuda -lcudart -lcurand)
# If compiling on an AMD platform, link HIP and rocRAND library (-lrocrand -L${HIP_ROOT_DIR}/lib -lhip_hcc)
```

Note: `${HIP_ROOT_DIR}` points to the directory where HIP was installed.
