# rocRAND

## The rocRAND repository is retired, please use the [ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries) repository

The rocRAND project provides functions that generate pseudorandom and quasirandom numbers.
The rocRAND library is implemented in the [HIP](https://github.com/ROCm/HIP)
programming language and optimized for AMD's latest discrete GPUs. It is designed to run on top
of AMD's [ROCm](https://rocm.docs.amd.com) runtime, but it also works on CUDA-enabled GPUs.

Prior to ROCm version 5.0, this project included the
[hipRAND](https://github.com/ROCm/hipRAND.git) wrapper. As of version 5.0, it was
split into a separate library. As of version 6.0, hipRAND can no longer be built from rocRAND.

## Requirements

* CMake (3.16 or later)
* C++ compiler with C++17 support to build the library.
  * Recommended to use at least gcc 9
  * clang uses the development headers and libraries from gcc, so a recent version of it must still
    be installed when compiling with clang
* C++ compiler with C++11 support to consume the library.
* For AMD platforms:
  * [ROCm](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/index.html) (1.7 or later)
  * [HIP-clang](https://github.com/ROCm/HIP/blob/master/INSTALL.md#hip-clang) compiler, which must be
    set as C++ compiler on ROCm platform.
* For CUDA platforms:
  * [HIP](https://github.com/ROCm/HIP)
  * Latest CUDA SDK
* Python 3.6 or higher (HIP on Windows only, only required for install script)
* Visual Studio 2019 with clang support (HIP on Windows only)
* Strawberry Perl (HIP on Windows only)

## Documentation

> [!NOTE]
> The published rocRAND documentation is available [here](https://rocm.docs.amd.com/projects/rocRAND/en/latest/) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).
