# rocRAND

The rocRAND project provides functions that generate pseudo-random and quasi-random numbers.

The rocRAND library is implemented in the [HIP](https://github.com/ROCm-Developer-Tools/HIP)
programming language and optimised for AMD's latest discrete GPUs. It is designed to run on top
of AMD's Radeon Open Compute [ROCm](https://rocm.github.io/) runtime, but it also works on
CUDA enabled GPUs.

Additionally, the project includes a wrapper library called hipRAND which allows user to easily port
CUDA applications that use cuRAND library to the [HIP](https://github.com/ROCm-Developer-Tools/HIP)
layer. In [ROCm](https://rocm.github.io/) environment hipRAND uses rocRAND, however in CUDA
environment cuRAND is used instead.

## Supported Random Number Generators

* XORWOW
* MRG32k3a
* Mersenne Twister for Graphic Processors (MTGP32)
* Philox (4x32, 10 rounds)
* Sobol32

## Requirements

* Git
* cmake (3.0.2 or later)
* C++ compiler with C++11 support
* For AMD platforms:
  * [ROCm](https://rocm.github.io/install.html) (1.7 or later)
  * [HCC](https://github.com/RadeonOpenCompute/hcc) compiler, which must be
    set as C++ compiler on ROCm platform.
* For CUDA platforms:
  * [HIP](https://github.com/ROCm-Developer-Tools/HIP) (hcc is not required)
  * Latest CUDA SDK

Optional:

* [GTest](https://github.com/google/googletest) (required only for tests; building tests is enabled by default)
  * Use `GTEST_ROOT` to specify GTest location (also see [FindGTest](https://cmake.org/cmake/help/latest/module/FindGTest.html))
  * Note: If GTest is not already installed, it will be automatically downloaded and built
* [TestU01](http://simul.iro.umontreal.ca/testu01/tu01.html) (required only for crush tests)
  * Use `TESTU01_ROOT_DIR` to specify TestU01 location
  * Note: If TestU01 is not already installed, it will be automatically downloaded and built
* Fortran compiler (required only for Fortran wrapper)
  * `gfortran` is recommended.
* Python 2.7+ or 3.5+ (required only for Python wrapper)

If some dependencies are missing, cmake script automatically downloads, builds and
installs them. Setting `DEPENDENCIES_FORCE_DOWNLOAD` option `ON` forces script to
not to use system-installed libraries, and to download all dependencies.

## Build and Install

```sh
git clone https://github.com/ROCmSoftwarePlatform/rocRAND.git

# Go to rocRAND directory, create and go to build directory
cd rocRAND; mkdir build; cd build

# Configure rocRAND, setup options for your system
# Build options: BUILD_TEST, BUILD_BENCHMARK (off by default), BUILD_CRUSH_TEST (off by default)
#
# ! IMPORTANT !
# On ROCm platform set C++ compiler to HCC. You can do it by adding 'CXX=<path-to-hcc>' or just
# `CXX=hcc` before 'cmake', or setting cmake option 'CMAKE_CXX_COMPILER' to path to the HCC compiler.
#
[CXX=hcc] cmake -DBUILD_BENCHMARK=ON ../. # or cmake-gui ../.

# Build
# For ROCM-1.6, if a HCC runtime error is caught, consider setting
# HCC_AMDGPU_TARGET=<arch> in front of make as a workaround
make -j4

# Optionally, run tests if they're enabled
ctest --output-on-failure

# Install
[sudo] make install
```

Note: Existing gtest library in the system (especially static gtest libraries built with other compilers)
may cause build failure; if errors are encountered with existing gtest library or other dependencies,
`DEPENDENCIES_FORCE_DOWNLOAD` flag can be passed to cmake, as mentioned before, to help solve the problem.

Note: To disable inline assembly optimisations in rocRAND (for both the host library and
the device functions provided in `rocrand_kernel.h`) set cmake option `ENABLE_INLINE_ASM`
to `OFF`.

## Running Unit Tests

```sh
# Go to rocRAND build directory
cd rocRAND; cd build

# To run all tests
ctest

# To run unit tests
./test/<unit-test-name>
```

## Running Benchmarks

```sh
# Go to rocRAND build directory
cd rocRAND; cd build

# To run benchmark for generate functions:
# engine -> all, xorwow, mrg32k3a, mtgp32, philox, sobol32
# distribution -> all, uniform-uint, uniform-float, uniform-double, normal-float, normal-double,
#                 log-normal-float, log-normal-double, poisson
# Further option can be found using --help
./benchmark/benchmark_rocrand_generate --engine <engine> --dis <distribution>

# To run benchmark for device kernel functions:
# engine -> all, xorwow, mrg32k3a, mtgp32, philox, sobol32
# distribution -> all, uniform-uint, uniform-float, uniform-double, normal-float, normal-double,
#                 log-normal-float, log-normal-double, poisson, discrete-poisson, discrete-custom
# further option can be found using --help
./benchmark/benchmark_rocrand_kernel --engine <engine> --dis <distribution>

# To compare against cuRAND (cuRAND must be supported):
./benchmark/benchmark_curand_generate --engine <engine> --dis <distribution>
./benchmark/benchmark_curand_kernel --engine <engine> --dis <distribution>
```

## Running Statistical Tests

```sh
# Go to rocRAND build directory
cd rocRAND; cd build

# To run "crush" test, which verifies that generated pseudorandom
# numbers are of high quality:
# engine -> all, xorwow, mrg32k3a, mtgp32, philox
./test/crush_test_rocrand --engine <engine>

# To run Pearson Chi-squared and Anderson-Darling tests, which verify
# that distribution of random number agrees with the requested distribution:
# engine -> all, xorwow, mrg32k3a, mtgp32, philox, sobol32
# distribution -> all, uniform-float, uniform-double, normal-float, normal-double,
#                 log-normal-float, log-normal-double, poisson
./test/stat_test_rocrand_generate --engine <engine> --dis <distribution>
```

## Using rocRAND with CMake

rocrand-config.cmake finds rocRAND include directory and library.

Use this module by invoking `find_package` with the form:

```cmake
find_package(rocrand
    [version] [EXACT]           # Minimum or EXACT version e.g. 0.5.0
    [REQUIRED]                  # Fail with error if rocRAND is not found
    CONFIG                      # Enable Config mode (might be required)
    [PATHS path1 [path2 ... ]]  # Path to install dir (e.g. /opt/rocm/rocrand)
)
```

The following variables are defined when rocRAND is found:

* rocrand_FOUND            - True if rocRAND headers and library were found.
* rocrand_INCLUDE_DIRS     - rocRAND include directory.
* rocrand_LIBRARY_DIRS     - Link directory for rocRAND library.
* rocrand_LIBRARIES        - rocRAND library to be linked.
* rocrand_VERSION          - ROCRAND_VERSION value from rocrand.h.

* rocrand_FORTRAN_FOUND    - True if rocRAND was built with Fortran wrapper; otherwise - false.
* rocrand_FORTRAN_SRC_DIRS - Directory with rocRAND Fortran wrapper sources rocrand_m.f90
                             and hip_m.f90. Include those source files, and add "use rocrand_m"
                             statement into your Fortran code.

Example (requires HIP):

```cmake
# First find and include HIP
# See https://github.com/ROCm-Developer-Tools/HIP
find_package(rocrand REQUIRED CONFIG PATHS "/opt/rocm/rocrand")

add_executable(foo foo.cpp)
# Set HIP flags, add HIP includes
# Link foo against HIP or CUDA lib (see hipconfig)
target_link_libraries(foo roc::rocrand)

# or using Fortran wrapper
set(bar_SRCS
    bar.f90
    ${rocrand_FORTRAN_SRC_DIRS}/hip_m.f90
    ${rocrand_FORTRAN_SRC_DIRS}/rocrand_m.f90
)
add_executable(bar ${bar_SRCS})
# Link bar against HIP or CUDA library (see hipconfig)
target_link_libraries(bar roc::rocrand)
```

## Using hipRAND with CMake

hiprand-config.cmake finds hipRAND include directory and library.

Use this module by invoking find_package with the form:

```cmake
find_package(hiprand
    [version] [EXACT]           # Minimum or EXACT version e.g. 0.5.0
    [REQUIRED]                  # Fail with error if hipRAND is not found
    CONFIG                      # Enable Config mode (might be required)
    [PATHS path1 [path2 ... ]]  # Path to install dir (e.g. /opt/rocm/hiprand)
)
```

The following variables are defined when hipRAND is found:

* hiprand_FOUND            - True if hipRAND headers and library were found.
* hiprand_INCLUDE_DIRS     - hipRAND include directory.
* hiprand_LIBRARY_DIRS     - Link directory for hipRAND library.
* hiprand_LIBRARIES        - hipRAND library to be linked.
* hiprand_VERSION          - HIPRAND_VERSION value from hiprand.h.

* hiprand_FORTRAN_FOUND    - True if hipRAND was built with Fortran wrapper; otherwise - false.
* hiprand_FORTRAN_SRC_DIRS - Directory with hipRAND Fortran wrapper sources hiprand_m.f90
                             and hip_m.f90. Include those source files, and add "use hiprand_m"
                             statement into your Fortran code.

Example (requires HIP, and rocRAND or cuRAND):

```cmake
# Remember to first find and include HIP
# See https://github.com/ROCm-Developer-Tools/HIP
find_package(hiprand REQUIRED CONFIG PATHS "/opt/rocm/hiprand")
if(HIP_PLATFORM STREQUAL "hcc")
    find_package(rocrand REQUIRED CONFIG PATHS "/opt/rocm/rocrand")
endif()

add_executable(foo foo.cpp)
# Set HIP flags, includes (see hipconfig)
if(HIP_PLATFORM STREQUAL "hcc")
    # Link foo against HIP library
    target_link_libraries(foo roc::rocrand hip::hiprand)
else()
    # Link foo against CUDA library
    target_link_libraries(foo "-lcurand" hip::hiprand)
endif()

# or using Fortran wrapper
set(bar_SRCS
    bar.f90
    ${hiprand_FORTRAN_SRC_DIRS}/hip_m.f90
    ${hiprand_FORTRAN_SRC_DIRS}/hiprand_m.f90
)
add_executable(bar ${bar_SRCS})
if(HIP_PLATFORM STREQUAL "hcc")
    # Link bar against HIP library
    target_link_libraries(bar roc::rocrand hip::hiprand)
else()
    # Link bar against CUDA libraries
    target_link_libraries(bar "-lcurand" hip::hiprand)
endif()
```

## Documentation

```sh
# go to rocRAND doc directory
cd rocRAND; cd doc

# run doxygen
doxygen Doxyfile

# open html/index.html

```

## Wrappers

* C++ wrappers for host API of rocRAND and hipRAND are in files [`rocrand.hpp`](./library/include/rocrand.hpp)
and [`hiprand.hpp`](./library/include/hiprand.hpp).
* [Fortran wrappers](./library/src/fortran/).
* [Python wrappers](./python/): [rocRAND](./python/rocrand) and [hipRAND](./python/hiprand).

## Support

Bugs and feature requests can be reported through [the issue tracker](https://github.com/ROCmSoftwarePlatform/rocRAND/issues).

## Contributions and License

Contributions of any kind are most welcome! More details are found at [CONTRIBUTING](./CONTRIBUTING.md)
and [LICENSE](./LICENSE.txt). Please note that [statistical tests](./test/crush) link to TestU01 library
distributed under GNU General Public License (GPL) version 3, thus GPL version 3 license applies to
that part of the project.
