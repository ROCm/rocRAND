# hipRAND

The hipRAND library provides functions that generate pseudo-random and quasi-random numbers.
The library is implemented in the [HIP](https://github.com/ROCm-Developer-Tools/HIP)
programming language, optimised for AMD's latest discrete GPUs. It is designed to run on top
of AMD's Radeon Open Compute [ROCm](https://rocm.github.io/) runtime, but it also works
on CUDA enabled GPUs.

Additionally, hipRAND includes a small wrapper library which allows user to easily port CUDA
applications that use cuRAND library to the [HIP](https://github.com/ROCm-Developer-Tools/HIP)
layer.

## Requirements

* cmake (2.8.12 or later)
* C++ compiler with C++11 support
* For AMD platforms:
    * [ROCm](https://rocm.github.io/install.html) (1.5 or later)
* For CUDA platforms:
    * [HIP](https://github.com/ROCm-Developer-Tools/HIP) (hcc is not required)
    * Latest CUDA SDK

## Build and install

```
git clone https://github.com/ROCmSoftwarePlatform/hipRAND.git

# go to hipRAND directory, create and go to build directory
cd hipRAND; mkdir build; cd build

# configure hipRAND, setup options for your system
cmake ../. # or cmake-gui ../.

# build
make -j4

# optionally, run tests if they're enabled
ctest

# install
sudo make install
```
