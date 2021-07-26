# Change Log for rocRAND

Full documentation for rocRAND is available at [https://rocrand.readthedocs.io/en/latest/](https://rocrand.readthedocs.io/en/latest/)

## (Unreleased) rocRAND-2.10.12
### Changed

- Packaging split into a runtime package called rocrand and a development package called rocrand-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.

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
