============
Installation
============

Introduction
------------

This chapter describes how to obtain rocRAND. There are two main methods: the easiest way is to install the prebuilt packages from the ROCm repositories. Alternatively, this chapter also describes how to build rocRAND from source.

Prebuilt Packages
-----------------

Installing the prebuilt rocRAND packages requires a ROCm-enabled platform. See the `ROCm documentation <https://docs.amd.com/>`_ for more information. After installing ROCm or enabling the ROCm repositories, rocRAND can be obtained using the system package manager.

For Ubuntu and Debian::

    sudo apt-get install rocrand

For CentOS::

    sudo yum install rocrand

For SLES::

    sudo dnf install rocrand

This will install rocRAND into the ``/opt/rocm`` directory.

Building rocRAND From Source
----------------------------

Obtaining Sources
^^^^^^^^^^^^^^^^^

The rocRAND sources are available from the `rocRAND GitHub Repository <https://github.com/ROCmSoftwarePlatform/rocRAND>`_. Use the branch that matches the system-installed version of ROCm. For example on a system that has ROCm 5.3 installed, use the following command to obtain rocRAND sources::

    git checkout -b rocm-5.3 https://github.com/ROCmSoftwarePlatform/rocRAND.git

Building The Library
^^^^^^^^^^^^^^^^^^^^

After obtaining the sources, rocRAND can be built using the installation script::

    cd rocrand
    ./install --install

This automatically builds all required dependencies, excluding HIP and Git, and installs the project to ``/opt/rocm`` if everything went well. See ``./install --help`` for further information.

Building With CMake
^^^^^^^^^^^^^^^^^^^

For a more elaborate installation process, rocRAND can be built manually using CMake. This enables certain configuration options that are not exposed to the ``./install`` script. In general, rocRAND can be built using CMake by configuring as follows::

    cd rocrand; mkdir build; cd build
    # Configure the project
    CXX=<compiler> cmake [options] ..
    # Build
    make -j4
    # Optionally, run the tests
    ctest --output-on-failure
    # Install
    [sudo] make install

To build for the ROCm platform,``<compiler>`` should be set to ``hipcc``. When building for CUDA or HIP-CPU, ``<compiler>`` should be set to the host compiler. If building for CUDA, then the location of ``nvcc`` may need to be passed explicitly using ``-DCMAKE_CUDA_COMPILER=<path-to-nvcc>`` if it is not on the path.

The following configuration options are available, in addition to the built-in CMake options:

* ``BUILD_FORTRAN_WRAPPER`` controls whether to build the Fortran wrapper. Defaults to ``OFF``.
* ``BUILD_TEST`` controls whether to build the rocRAND tests. Defaults to ``OFF``.
* ``BUILD_BENCHMARK`` controls whether to build the rocRAND benchmarks. Defaults to ``OFF``.
* ``BUILD_ADDRESS_SANITIZER`` controls whether to build with address sanitization enabled. Defaults to ``OFF``.
* ``USE_HIP_CPU`` is an experimental option that controls whether to build for HIP-CPU. Defaults to ``OFF``.

To install rocRAND with a non-standard installation location of ROCm, pass ``-DCMAKE_PREFIX_PATH=</path/to/opt/rocm/>`` or set the environment variable ``ROCM_PATH`` to ``path/to/opt/rocm``.

Building the Python API Wrapper
-------------------------------

Requirements
^^^^^^^^^^^^

The rocRAND Python API Wrapper requires the following dependencies:

* rocRAND
* Python 3.5
* NumPy (will be installed automatically as a dependency if necessary)

Note: If rocRAND is built from sources but not installed or installed in
non-standard directory, set the ``ROCRAND_PATH`` environment variable. For example::

    export ROCRAND_PATH=~/rocRAND/build/library/

Installing
^^^^^^^^^^

The Python rocRAND module can be installed using pip::

    cd rocrand/python/rocrand
    pip install .

The tests can be executed as follows::

    cd rocrand/python/rocrand
    python tests/rocrand_test.py

