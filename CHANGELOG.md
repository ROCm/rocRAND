# Change Log for rocRAND

Full documentation for rocRAND is available at [https://rocrand.readthedocs.io/en/latest/](https://rocrand.readthedocs.io/en/latest/)

## (Unreleased) rocRAND for ROCm 5.0.0
### Changed
- [hipRAND](https://github.com/ROCmSoftwarePlatform/hipRAND.git) split into a separate package
- Header file installation location changed to match other libraries.
  - Using the `rocrand.h` header file should now use `#include <rocrand/rocrand.h>`, rather than `#include <rocrand/rocrand.h>`
- rocRAND still includes hipRAND using a submodule
  - The rocRAND package also sets the provides field with hipRAND, so projects which require hipRAND can begin to specify it.

### Added
- Generating a random sequence different sizes now produces the same sequence without gaps
  indepent of how many values are generated per call.
  - Only in the case of XORWOW and SOBOL32
  - This only holds true if the size in each call is a divisor of the distributions
    `output_width` due to performance
  - Similarly the output pointer has to be aligned to `output_width * sizeof(output_type)`

### Fixed
- Fix offset behaviour for XORWOW generator, setting offset now correctly generates the same sequence
starting from the offset.
  - Only uniform int and float will work as these can be generated with a single call to the generator

## (Unreleased) rocRAND-2.10.12 for ROCm 4.5.0
### Addded
- Initial HIP on Windows support. See README for instructions on how to build and install.
### Changed
- Packaging split into a runtime package called rocrand and a development package called rocrand-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.
### Fixed
- Fix for mrg_uniform_distribution_double generating incorrect range of values
- Fix for order of state calls for log_normal, normal, and uniform
### Known issues
- kernel_xorwow test is failing for certain GPU architectures.

## [Unreleased rocRAND-2.10.11 for ROCm 4.4.0]
### Added
- Sobol64 support added.
- Benchmark time measurement improvement
- Address Sanitizer build option added.
### Fixed
- nvcc backend fix
- Fix ranges of MRG32k3a device functions.

## [Unreleased rocRAND-2.10.10 for ROCm 4.3.0]
### Added
- gfx90a support added.
- gfx1030 support added
- gfx803 supported re-enabled
### Fixed
- Memory leaks in Poisson tests has been fixed.
- Memory leaks when generator has been created but setting seed/offset/dimensions throws an exception has been fixed.

## [rocRAND-2.10.9 for ROCm 4.2.0]
### Fixed
- rocRAND benchmark performance drop for xorwow has been fixed for older ROCm builds.

## [rocRAND-2.10.8 for ROCm 4.1.0]
### Added
- Ability to force install dependencies with new -d flag in install script
### Changed
- rocRAND package name has been updated to support newer versions of ROCm.
### Fixed
- rocRAND benchmark performance drop has been fixed.
- Debug builds via the install script have been fixed.

## [rocRAND-2.10.7 for ROCm 4.0.0]
### Added
- No new features

## [rocRAND-2.10.6 for ROCm 3.10]
### Added
- No new features

## [rocRAND-2.10.5 for ROCm 3.9.0]
### Added
- No new features

## [rocRAND-2.10.4 for ROCm 3.8.0]
### Added
- No new features

## [rocRAND-2.10.3 for ROCm 3.7.0]
### Fixed
- Fixed package naming to reflect OS name and architecture.

## [rocRAND-2.10.2 for ROCm 3.6.0]
### Added
- No new features

## [rocRAND-2.10.1 for ROCm 3.5.0]
### Added
- Static library build options added in beta (subject to change in build method and naming in future releases)
### Changed
- Switched to hip-clang as default compiler
### Deprecated
- HCC build deprecated
