# Changelog for rocRAND

Documentation for rocRAND is available at
[https://rocm.docs.amd.com/projects/rocRAND/en/latest/](https://rocm.docs.amd.com/projects/rocRAND/en/latest/)

## (Unreleased) rocRAND-3.1.0 for ROCm 6.2.0

### Additions

* Added `rocrand_create_generator_host`
  * The following generators are supported:
    * `ROCRAND_RNG_PSEUDO_MRG31K3P`
    * `ROCRAND_RNG_PSEUDO_MRG32K3A`
    * `ROCRAND_RNG_PSEUDO_PHILOX4_32_10`
    * `ROCRAND_RNG_PSEUDO_THREEFRY2_32_20`
    * `ROCRAND_RNG_PSEUDO_THREEFRY2_64_20`
    * `ROCRAND_RNG_PSEUDO_THREEFRY4_32_20`
    * `ROCRAND_RNG_PSEUDO_THREEFRY4_64_20`
    * `ROCRAND_RNG_PSEUDO_XORWOW`
    * `ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32`
    * `ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64`
    * `ROCRAND_RNG_QUASI_SOBOL32`
    * `ROCRAND_RNG_QUASI_SOBOL64`
  * The host-side generators support multi-core processing. On Linux, this requires the TBB (Thread Building Blocks) development package to be installed on the system when building rocRAND (`libtbb-dev` on Ubuntu and derivatives).
    * If TBB is not found when configuring rocRAND, the configuration is still successful, and the host generators are executed on a single CPU thread.
* Added the option to create a host generator to the Python wrapper
* Added the option to create a host generator to the Fortran wrapper
* Added dynamic ordering. This ordering is free to rearrange the produced numbers, 
  which can be specific to devices and distributions. It is implemented for:
  * XORWOW, MRG32K3A, MTGP32, Philox 4x32-10, MRG31K3P, LFSR113, and ThreeFry
* For the NVIDIA platform compilation using clang as the host compiler is now supported.
* C++ wrapper:
  * `lfsr113_engine` now also supports being constructed with a seed of type `unsigned long long`, not only `uint4`.
  * added optional order parameter to constructor of `mt19937_engine`
* Added the following functions for the `ROCRAND_RNG_PSEUDO_MTGP32` generator:
  * `rocrand_normal2`
  * `rocrand_normal_double2`
  * `rocrand_log_normal2`
  * `rocrand_log_normal_double2`
* Added `rocrand_create_generator_host_blocking` which dispatches without stream semantics.
* Added host-side generator for `ROCRAND_RNG_PSEUDO_MTGP32`.
* Added offset and skipahead functionality to LFSR113 generator.
* Added dynamic ordering for architecture `gfx1102`.

### Changes

* For device-side generators, you can now wrap calls to rocrand_generate_* inside of a hipGraph. There are a few 
  things to be aware of:
  - Generator creation (rocrand_create_generator), initialization (rocrand_initialize_generator), and destruction (rocrand_destroy_generator) must still happen outside the hipGraph.
  - After the generator is created, you may call API functions to set its seed, offset, and order.
  - After the generator is initialized (but before stream capture or manual graph creation begins), use rocrand_set_stream to set the stream the generator will use within the graph.
  - A generator's seed, offset, and stream may not be changed from within the hipGraph. Attempting to do so may result in unpredicable behaviour.
  - API calls for the poisson distribution (eg. rocrand_generate_poisson) are not yet supported inside of hipGraphs.
  - For sample usage, see the unit tests in test/test_rocrand_hipgraphs.cpp
* Building rocRAND now requires a C++17 capable compiler, as the internal library sources now require it. However consuming rocRAND is still possible from C++11 as public headers don't make use of the new features.
* Building rocRAND should be faster on machines with multiple CPU cores as the library has been
  split to multiple compilation units.
* C++ wrapper: the `min()` and `max()` member functions of the generators and distributions are now `static constexpr`.
* Rename and unify the existing ROCRAND_DETAIL_.*_BM_NOT_IN_STATE to ROCRAND_DETAIL_BM_NOT_IN_STATE
* Static & dynamic library: moved all internal symbols to namespaces to avoid potential symbol name collisions when linking.

### Deprecations

* Deprecated the following typedefs. Please use the unified `state_type` alias instead.
  * `rocrand_device::threefry2x32_20_engine::threefry2x32_20_state`
  * `rocrand_device::threefry2x64_20_engine::threefry2x64_20_state`
  * `rocrand_device::threefry4x32_20_engine::threefry4x32_20_state`
  * `rocrand_device::threefry4x64_20_engine::threefry4x64_20_state`
* Deprecated internal header: src/rng/distribution/distributions.hpp
* Deprecated internal header: src/rng/device_engines.hpp

### Removals

* Removed references to and workarounds for deprecated hcc.
* Support for HIP-CPU

### Known issues
- SOBOL64 and SCRAMBLED_SOBOL64 generate poisson-distributed `unsigned long long int` numbers instead of `unsigned int`. This will be fixed in the next major release.

## (Unreleased) rocRAND-3.0.0 for ROCm 6.0.0

### Additions

* Added `rocrand_create_generator_host` with initial support for `ROCRAND_RNG_PSEUDO_PHILOX4_32_10` and `ROCRAND_RNG_PSEUDO_MRG31K3P`.
* Added the option to create a host generator to the Python wrapper
* Added the option to create a host generator to the Fortran wrapper

### Changes

* Generator classes from `rocrand.hpp` are no longer copyable (in previous versions these copies
  would copy internal references to the generators and would lead to double free or memory leak
  errors)
  * These types should be moved instead of copied; move constructors and operators are now
    defined

### Optimizations

* Improved MT19937 initialization and generation performance

### Removals

* Removed the hipRAND submodule from rocRAND; hipRAND is now only available as a separate
  package
* Removed references to, and workarounds for, the deprecated hcc

### Fixes

* `mt19937_engine` from `rocrand.hpp` is now move-constructible and move-assignable (the move
  constructor and move assignment operator was deleted for this class)
* Various fixes for the C++ wrapper header `rocrand.hpp`
  * The name of `mrg31k3p` it is now correctly spelled (was incorrectly named `mrg31k3a` in previous
    versions)
  * Added the missing `order` setter method for `threefry4x64`
  * Fixed the default ordering parameter for `lfsr113`
* Build error when using Clang++ directly resulting from unsupported `amdgpu-target` references
* Added hip::device as dependency to benchmark_rocrand_tuning to make it compile with amdclang++.
* Minor entropy waste in 64-bits Threefry function producing two log-normally-distributed doubles.

## rocRAND-2.10.17 for ROCm 5.5.0

### Additions

* MT19937 pseudo random number generator based on M. Matsumoto and T. Nishimura, 1998,
  *Mersenne Twister: A 623-dimensionally equidistributed uniform pseudorandom number generator*
* New benchmark APIs for Google Benchmark:
  * `benchmark_rocrand_device_api` replaces `benchmark_rocrand_kernel`
  * `benchmark_curand_host_api` replaces `benchmark_curand_generate`
  * `benchmark_curand_device_api` replaces `benchmark_curand_kernel`
* Experimental HIP-CPU feature
* ThreeFry pseudorandom number generator based on Salmon et al., 2011, *Parallel random numbers:
  as easy as 1, 2, 3*
* Accessor methods for SOBOL 32 and 64 direction vectors and constants:
  * Enum `rocrand_direction_vector_set` to select the direction vector set
  * `rocrand_get_direction_vectors32(...)` supersedes:
    * `rocrand_h_sobol32_direction_vectors`
    * `rocrand_h_scrambled_sobol32_direction_vectors`
  * `rocrand_get_direction_vectors64(...)` supersedes:
    * `rocrand_h_sobol64_direction_vectors`
    * `rocrand_h_scrambled_sobol64_direction_vectors`
  * `rocrand_get_scramble_constants32(...)` supersedes `h_scrambled_sobol32_constants`
  * `rocrand_get_scramble_constants64(...)` supersedes `h_scrambled_sobol64_constants`

### Changes

* Python 2.7 is no longer officially supported

## rocRAND-2.10.16 for ROCm 5.4.0

### Additions

* MRG31K3P pseudorandom number generator based on L'Ecuyer and Touzin, 2000, *Fast combined
  multiple recursive generators with multipliers of the form a = ±2q ±2r*
* LFSR113 pseudorandom number generator based on L'Ecuyer, 1999, *Tables of maximally
  equidistributed combined LFSR generators*
* `SCRAMBLED_SOBOL32` and `SCRAMBLED_SOBOL64` quasirandom number generators (scrambled
  Sobol sequences are generated by scrambling the output of a Sobol sequence)

### Changes

* The `mrg_<distribution>_distribution` structures, which provide numbers based on MRG32K3A, have been replaced by `mrg_engine_<distribution>_distribution`, where `<distribution>` is `log_normal`, `normal`, `poisson`, or `uniform`
  * These structures provide numbers for MRG31K3P (with template type `rocrand_state_mrg31k3p`)
    and MRG32K3A (with template type `rocrand_state_mrg32k3a`)

### Fixes

* Sobol64 now returns 64-bit (instead of 32-bit) random numbers, which results in the performance of
  this generator being regressed
* Bug that prevented Windows code compilation in C++ mode (with a host compiler) when rocRAND
  headers were included

## rocRAND-2.10.15 for ROCm 5.3.0

### Additions

* New benchmark for the host API using Google benchmark that replaces
  `benchmark_rocrand_generate`, which is deprecated

### Changes

* Increased the number of warmup iterations for `rocrand_benchmark_generate` from 5 to 15 to
  eliminate corner cases that generate artificially high benchmark scores

## (Released) rocRAND-2.10.14 for ROCm 5.2.0

### Additions

* Backward compatibility for `#include <rocrand.h>` (deprecated) using wrapper header files
* Packages for test and benchmark executables on all supported operating systems using CPack

## rocRAND-2.10.13 for ROCm 5.1.0

### Additions

* Generating a random sequence of different sizes now produces the sequence without gaps,
    independent of how many values are generated per call
  * This is only in the case of XORWOW, MRG32K3A, PHILOX4X32_10, SOBOL32, and SOBOL64
  * This is only true if the size in each call is a divisor of the distributions `output_width` due to
    performance
  * The output pointer must be aligned with `output_width * sizeof(output_type)`

### Changes

* [hipRAND](https://github.com/ROCmSoftwarePlatform/hipRAND.git) has been split into a separate
  package
* Header file installation location changed to match other libraries.
  * When using the `rocrand.h` header file, use `#include <rocrand/rocrand.h>` rather than
    `#include <rocrand.h>`
* rocRAND still includes hipRAND using a submodule
  * The rocRAND package sets the provides field with hipRAND, so projects that require hipRAND can
    begin to specify it

### Fixes

* Offset behavior for XORWOW, MRG32K3A, and PHILOX4X32_10 generator
  * Setting offset now correctly generates the same sequence starting from the offset
  * Only uniform `int` and `float` will work, as these can be generated with a single call to the generator

### Known issues

* `kernel_xorwow` unit test is failing for certain GPU architectures

## rocRAND-2.10.12 for ROCm 5.0.0

There are no updates for this ROCm release.

## rocRAND-2.10.12 for ROCm 4.5.0

### Additions

* Initial HIP on Windows support

### Changes

* Packaging has been split into a runtime package (`rocrand`) and a development package
  (`rocrand-devel`):
  The development package depends on the runtime package. When installing the runtime package,
  the package manager will suggest the installation of the development package to aid users
  transitioning from the previous version's combined package. This suggestion by package manager is
  for all supported operating systems (except CentOS 7) to aid in the transition. The `suggestion`
  feature in the runtime package is introduced as a deprecated feature and will be removed in a future
  ROCm release.

### Fixes

* `mrg_uniform_distribution_double` is no longer generating an incorrect range of values
* Order of state calls for `log_normal`, `normal`, and `uniform`

### Known issues

* `kernel_xorwow` test is failing for certain GPU architectures

## [rocRAND-2.10.11 for ROCm 4.4.0]

### Additions

* Sobol64 support
* Benchmark time measurement improvement
* AddressSanitizer build option

### Fixes

* NVCC backend fix
* Fix ranges of MRG32k3a device functions

## [rocRAND-2.10.10 for ROCm 4.3.0]

### Additions

* gfx90a support
* gfx1030 support
* gfx803 supported re-enabled

### Fixes

* Memory leaks in Poisson tests
* Memory leaks when generator is created, but setting seed/offset/dimensions throws an exception

## [rocRAND-2.10.9 for ROCm 4.2.0]

### Fixes

* The rocRAND benchmark performance drop for `xorwow` has been fixed for older ROCm builds

## [rocRAND-2.10.8 for ROCm 4.1.0]

### Additions

* Ability to force install dependencies with new `-d` flag in install script

### Changes

* rocRAND package name has been updated to support newer versions of ROCm

### Fixes

* rocRAND benchmark performance drop
* Debug builds via the install script

## [rocRAND-2.10.7 for ROCm 4.0.0]

There are no updates for this ROCm release.

## [rocRAND-2.10.6 for ROCm 3.10]

There are no updates for this ROCm release.

## [rocRAND-2.10.5 for ROCm 3.9.0]

There are no updates for this ROCm release.

## [rocRAND-2.10.4 for ROCm 3.8.0]

There are no updates for this ROCm release.

## [rocRAND-2.10.3 for ROCm 3.7.0]

### Fixes

- Package naming now reflects operating system name and architecture

## [rocRAND-2.10.2 for ROCm 3.6.0]

There are no updates for this ROCm release.

## [rocRAND-2.10.1 for ROCm 3.5.0]

### Additions

- Static library build options were added in the beta; these are subject to change (build method and
  naming) in future releases

### Changes

- HIP-Clang is now the default compiler

### Deprecations

- HCC build
