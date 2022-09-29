# rocRAND

The rocRAND project provides functions that generate pseudo-random and quasi-random numbers.

The rocRAND library is implemented in the [HIP](https://github.com/ROCm-Developer-Tools/HIP)
programming language and optimised for AMD's latest discrete GPUs. It is designed to run on top
of AMD's Radeon Open Compute [ROCm](https://rocm.github.io/) runtime, but it also works on
CUDA enabled GPUs.

Prior to ROCm version 5.0, this project included the [hipRAND](https://github.com/ROCmSoftwarePlatform/hipRAND.git) wrapper. As of version 5.0, this has been split into a separate library.

## Supported Random Number Generators

* XORWOW
* MRG31k3p
* MRG32k3a
* Mersenne Twister for Graphic Processors (MTGP32)
* Philox (4x32, 10 rounds)
* LFSR113
* Sobol32
* Scrambled Sobol32
* Sobol64
* Scrambled Sobol64

## Requirements

* CMake (3.16 or later)
* C++ compiler with C++11 support
* For AMD platforms:
  * [ROCm](https://rocm.github.io/install.html) (1.7 or later)
  * [HIP-clang](https://github.com/ROCm-Developer-Tools/HIP/blob/master/INSTALL.md#hip-clang) compiler, which must be
    set as C++ compiler on ROCm platform.
* For CUDA platforms:
  * [HIP](https://github.com/ROCm-Developer-Tools/HIP)
  * Latest CUDA SDK
* Python 3.6 or higher (HIP on Windows only, only required for install script)
* Visual Studio 2019 with clang support (HIP on Windows only)
* Strawberry Perl (HIP on Windows only)

Optional:

* [GTest](https://github.com/google/googletest) (required only for tests; building tests is enabled by default)
  * Use `GTEST_ROOT` to specify GTest location (also see [FindGTest](https://cmake.org/cmake/help/latest/module/FindGTest.html))
  * Note: If GTest is not already installed, it will be automatically downloaded and built
* Fortran compiler (required only for Fortran wrapper)
  * `gfortran` is recommended.
* Python 2.7+ or 3.5+ (required only for Python wrapper)

If some dependencies are missing, cmake script automatically downloads, builds and
installs them. Setting `DEPENDENCIES_FORCE_DOWNLOAD` option `ON` forces script to
not to use system-installed libraries, and to download all dependencies.

## Build and Install

```
git clone https://github.com/ROCmSoftwarePlatform/rocRAND.git

# Go to rocRAND directory, create and go to build directory
cd rocRAND; mkdir build; cd build

# Configure rocRAND, setup options for your system
# Build options: BUILD_TEST (off by default), BUILD_BENCHMARK (off by default), BUILD_SHARED_LIBS (on by default)
#
# ! IMPORTANT !
# Set C++ compiler to HIP-clang. You can do it by adding 'CXX=<path-to-compiler>'
# before 'cmake' or setting cmake option 'CMAKE_CXX_COMPILER' to path to the compiler.
#
# The python interface do not work with static library.
#
[CXX=hipcc] cmake -DBUILD_BENCHMARK=ON ../. # or cmake-gui ../.

# To configure rocRAND for Nvidia platforms, 'CXX=<path-to-nvcc>', `CXX=nvcc` or omitting the flag
# entirely before 'cmake' is sufficient
[CXX=nvcc] cmake -DBUILD_BENCHMARK=ON ../. # or cmake-gui ../.
# or
cmake -DBUILD_BENCHMARK=ON ../. # or cmake-gui ../.

# Build
make -j4

# Optionally, run tests if they're enabled
ctest --output-on-failure

# Install
[sudo] make install
```

### HIP on Windows

Initial support for HIP on Windows has been added.  To install, use the provided rmake.py python script:
```shell
git clone https://github.com/ROCmSoftwarePlatform/rocRAND.git
cd rocRAND

# the -i option will install rocPRIM to C:\hipSDK by default
python rmake.py -i

# the -c option will build all clients including unit tests
python rmake.py -c
```

Note: Existing gtest library in the system (especially static gtest libraries built with other compilers)
may cause build failure; if errors are encountered with existing gtest library or other dependencies,
`DEPENDENCIES_FORCE_DOWNLOAD` flag can be passed to cmake, as mentioned before, to help solve the problem.

Note: To disable inline assembly optimisations in rocRAND (for both the host library and
the device functions provided in `rocrand_kernel.h`) set cmake option `ENABLE_INLINE_ASM`
to `OFF`.

## Running Unit Tests

```
# Go to rocRAND build directory
cd rocRAND; cd build

# To run all tests
ctest

# To run unit tests
./test/<unit-test-name>
```

## Running Benchmarks

```
# Go to rocRAND build directory
cd rocRAND; cd build

# To run benchmark for the host generate functions:
# The benchmarks are registered with Google Benchmark as `device_generate<engine,distribution>`, where
# engine -> xorwow, mrg31k3p, mrg32k3a, mtgp32, philox, lfsr113, 
#           sobol32, scrambled_sobol32, sobol64, scrambled_sobol64
# distribution -> uniform-uint, uniform-uchar, uniform-ushort, 
#                 uniform-half, uniform-float, uniform-double, 
#                 normal-half, normal-float, normal-double,
#                 log-normal-half, log-normal-float, log-normal-double, poisson
# Further option can be found using --help
./benchmark/benchmark_rocrand_host_api
# To run specific benchmarks:
./benchmark/benchmark_rocrand_host_api --benchmark_filter=<regex>
# For example to run benchmarks with engine sobol64:
./benchmark_rocrand_host_api --benchmark_filter="device_generate<sobol64*"
# To view all registered benchmarks:
./benchmark_rocrand_host_api --benchmark_list_tests=true
# The benchmark also supports user input:
./benchmark_rocrand_host_api --size <number> --trials <number> --offset <number> --dimensions <number> --lambda <float float float ...>

# To run benchmark for device kernel functions:
# engine -> all, xorwow, mrg31k3p, mrg32k3a, mtgp32, philox, lfsr113, 
#           sobol32, scrambled_sobol32, sobol64, scrambled_sobol64
# distribution -> all, uniform-uint, uniform-float, uniform-double, normal-float, normal-double,
#                 log-normal-float, log-normal-double, poisson, discrete-poisson, discrete-custom
# further option can be found using --help
./benchmark/benchmark_rocrand_kernel --engine <engine> --dis <distribution>

# To compare against cuRAND (cuRAND must be supported):
./benchmark/benchmark_curand_generate --engine <engine> --dis <distribution>
./benchmark/benchmark_curand_kernel --engine <engine> --dis <distribution>
```

### Legacy benchmarks

The legacy benchmarks (before the move to using googlebenchmark) can be disabled by setting the
cmake option `BUILD_LEGACY_BENCHMARK` to `OFF`. For compatibility, this settings defaults to `ON`
when `BUILD_BENCHMARK` is set.
The legacy benchmarks are deprecated and will be removed in a future version once all benchmarks are
migrated to the new framework.

## Running Statistical Tests

```
# Go to rocRAND build directory
cd rocRAND; cd build

# To run Pearson Chi-squared and Anderson-Darling tests, which verify
# that distribution of random number agrees with the requested distribution:
# engine -> all, xorwow, mrg31k3p, mrg32k3a, mtgp32, philox, lfsr113, 
#           sobol32, scrambled_sobol32, sobol64, scrambled_sobol64
# distribution -> all, uniform-float, uniform-double, normal-float, normal-double,
#                 log-normal-float, log-normal-double, poisson
./test/stat_test_rocrand_generate --engine <engine> --dis <distribution>
```

## Documentation
The latest rocRAND documentation and API description can be found [here](https://rocrand.readthedocs.io/en/latest/).

It can also be build using the following commands
```
# go to rocRAND doc directory
cd rocRAND; cd doc

# run doxygen
doxygen Doxyfile

# open html/index.html

```

## Wrappers

* C++ wrappers for host API of rocRAND and hipRAND are in files [`rocrand.hpp`](./library/include/rocrand/rocrand.hpp)
and [`hiprand.hpp`](./library/include/rocrand/hiprand.hpp).
* [Fortran wrappers](./library/src/fortran/).
* [Python wrappers](./python/): [rocRAND](./python/rocrand) and [hipRAND](./python/hiprand).

## Support

Bugs and feature requests can be reported through [the issue tracker](https://github.com/ROCmSoftwarePlatform/rocRAND/issues).

## Contributions and License

Contributions of any kind are most welcome! More details are found at [CONTRIBUTING](./CONTRIBUTING.md)
and [LICENSE](./LICENSE.txt). Please note that [statistical tests](./test/crush) link to TestU01 library
distributed under GNU General Public License (GPL) version 3, thus GPL version 3 license applies to
that part of the project.
