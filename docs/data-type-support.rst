Data type support
******************************************

Host API
========

Generator types
---------------
 .. list-table:: Supported generators
    :header-rows: 1
    :name: supported-generators

    *
      - Generator
      - rocRAND support
      - cuRAND support
    *
      - XORWOW
      - ✅
      - ✅
    *
      - MRG32K3A
      - ✅
      - ✅
    *
      - MTGP32
      - ✅
      - ✅
    *
      - Philox 4x32-10
      - ✅
      - ✅
    *
      - MT19937
      - ✅
      - ✅
    *
      - MRG31K3P
      - ✅
      - ❌
    *
      - LFSR113
      - ✅
      - ❌
    *
      - ThreeFry 2x32-20
      - ✅
      - ❌
    *
      - ThreeFry 4x32-20
      - ✅
      - ❌
    *
      - ThreeFry 2x64-20
      - ✅
      - ❌
    *
      - ThreeFry 4x64-20
      - ✅
      - ❌
    *
      - Sobol32
      - ✅
      - ✅
    *
      - Scrambled Sobol32
      - ✅
      - ✅
    *
      - Sobol64
      - ✅
      - ✅
    *
      - Scrambled Sobol64
      - ✅
      - ✅

Only Sobol64, Scrambled Sobol64, ThreeFry 2x64-20 and ThreeFry 4x64-20 support generation of 64 bit :code:`unsigned long long` integers.

Seed types
----------

All generators can be seeded with :code:`unsigned long long`, however LFSR113 can additionally be seeded using an :code:`uint4`.

Output types
------------

The generators produce random numbers chosen from a given distribution. The following distributions and corresponding output types are supported for the host API:

Uniform distribution
""""""""""""""""""""

 .. list-table:: Supported types for uniform distributions
    :header-rows: 1
    :name: host-types-uniform-distribution

    *
      - Type
      - Size of type
      - rocRAND support
      - cuRAND support
    *
      - unsigned char
      - 8 bit
      - ✅
      - ❌
    *
      - unsigned short
      - 16 bit
      - ✅
      - ❌
    *
      - unsigned int
      - 32 bit
      - ✅
      - ✅
    *
      - unsigned long long
      - 64 bit [#]_
      - ✅
      - ✅
    *
      - half
      - 16 bit
      - ✅
      - ❌
    *
      - float
      - 32 bit
      - ✅
      - ✅
    *
      - double
      - 64 bit
      - ✅
      - ✅

Uniform distributions of integral types return a number between 0 and 2^(size in bits) - 1, whereas floating-point types return a number between 0.0 and 1.0, excluding 1.0.
Only Sobol64, Scrambled Sobol64, ThreeFry 2x64-20 and ThreeFry 4x64-20 support generation of 64 bit `unsigned long long` integers.

Poisson distribution
"""""""""""""""""""""

 .. list-table:: Supported types for the poisson distribution
    :header-rows: 1
    :name: host-types-normal-distribution

    *
      - Type
      - Size of type
      - rocRAND support
      - cuRAND support
    *
      - unsigned int
      - 32 bit
      - ✅
      - ✅

Normal distribution
"""""""""""""""""""""

 .. list-table:: Supported types for normal distributions
    :header-rows: 1
    :name: host-types-normal-distribution

    *
      - Type
      - Size of type
      - rocRAND support
      - cuRAND support
    *
      - half
      - 16 bit
      - ✅
      - ❌
    *
      - float
      - 32 bit
      - ✅
      - ✅
    *
      - double
      - 64 bit
      - ✅
      - ✅

Log-normal distributions
"""""""""""""""""""""

 .. list-table:: Supported types for log-normal distributions
    :header-rows: 1
    :name: host-types-log-normal-distribution

    *
      - Type
      - Size of type
      - rocRAND support
      - cuRAND support
    *
      - half
      - 16 bit
      - ✅
      - ❌
    *
      - float
      - 32 bit
      - ✅
      - ✅
    *
      - double
      - 64 bit
      - ✅
      - ✅

Device API
==========

Generator types
---------------
 .. list-table:: Supported generators
    :header-rows: 1
    :name: supported-generators

    *
      - Generator
      - rocRAND support
      - cuRAND support
      - native size
    *
      - XORWOW
      - ✅
      - ✅
    *
      - MRG32K3A
      - ✅
      - ✅
    *
      - MTGP32
      - ✅
      - ✅
    *
      - Philox 4x32-10
      - ✅
      - ✅
    *
      - MT19937
      - ❌
      - ❌
    *
      - MRG31K3P
      - ✅
      - ❌
    *
      - LFSR113
      - ✅
      - ❌
    *
      - ThreeFry 2x32-20
      - ✅
      - ❌
    *
      - ThreeFry 4x32-20
      - ✅
      - ❌
    *
      - ThreeFry 2x64-20
      - ✅
      - ❌
    *
      - ThreeFry 4x64-20
      - ✅
      - ❌
    *
      - Sobol32
      - ✅
      - ✅
    *
      - Scrambled Sobol32
      - ✅
      - ✅
    *
      - Sobol64
      - ✅
      - ✅
    *
      - Scrambled Sobol64
      - ✅
      - ✅

Seed types
----------

All generators can be seeded with :code:`unsigned long long`, however LFSR113 can additionally be seeded using an :code:`uint4`.

Output types
------------

The generators produce random numbers chosen from a given distribution. The following distributions and corresponding output types are supported for the device API, however not all generators support all types:


Uniform distribution
""""""""""""""""""""

 .. list-table:: Supported types for uniform distributions
    :header-rows: 1
    :name: device-types-uniform-distribution

    *
      - Type
      - rocRAND support
      - supported rocRAND generators
      - cuRAND support
    *
      - unsigned int
      - ✅
      - all native 32-bit generators
      - ✅
    *
      - unsigned long long int
      - ✅
      - all native 64-bit generators
      - ✅
    *
      - float
      - ✅
      - all generators
      - ✅
    *
      - float2
      - ✅
      - Philox 4x32-10
      - ❌
    *
      - float4
      - ✅
      - Philox 4x32-10
      - ✅
    *
      - double
      - ✅
      - all generators
      - ✅
    *
      - double2
      - ✅
      - Philox 4x32-10
      - ✅
    *
      - double4
      - ✅
      - Philox 4x32-10
      - ❌


Normal distribution
""""""""""""""""""""

 .. list-table:: Supported types for normal distributions
    :header-rows: 1
    :name: device-types-normal-distribution

    *
      - Type
      - rocRAND support
      - supported rocRAND generators
      - cuRAND support
    *
      - float
      - ✅
      - all generators
      - ✅
    *
      - double
      - ✅
      - all generators
      - ✅
    *
      - float2
      - ✅
      - Philox 4x32-10, mrg31k3p, mrg32k3a, xorwow, lfsr113, all threefry generators
      NOT: mtgp32, sobol32, scrambled sobol32, sobol64, scrambled sobol64
      TODO: why do these generators not support float2 or double2?
      TODO: Why is `half` not supported?
      -> There are functions for both in the detail namespace, but they are not used anywhere afaik
        -> they are used in the cpp wrapper? Or what the hell is happening?
      - ✅
    *
      - float4
      - ✅
      - Philox 4x32-10
      - ✅
    *
      - double2
      - ✅
      - Philox 4x32-10, mrg31k3p, mrg32k3a, xorwow, lfsr113, all threefry generators
      NOT: mtgp32, sobol32, scrambled sobol32, sobol64, scrambled sobol64
      - ✅
    *
      - double4
      - ✅
      - Philox 4x32-10
      - ❌

Log-normal distributions
"""""""""""""""""""""

 .. list-table:: Supported types for log-normal distributions
    :header-rows: 1
    :name: device-types-log-normal-distribution

    *
      - Type
      - rocRAND support
      - supported rocRAND generators
      - cuRAND support
    *
      - half
      - all generators
      - ❌
      - ❌
    *
      - float
      - all generators
      - ✅
      - ✅
    *
      - double
      - all generators
      - ✅
      - ✅
    *
      - float2
      - Philox 4x32-10, mrg31k3p, mrg32k3a, xorwow, lfsr113, all threefry generators
      NOT: mtgp32, sobol32, scrambled sobol32, sobol64, scrambled sobol64
      - ✅
      - ✅
    *
      - double2
      - philox 4x32-10, mrg31k3p, mrg32k3a, xorwow, lfsr113, all threefry generators
      NOT: mtgp32, sobol32, scrambled sobol32, sobol64, scrambled sobol64
      TODO: why not the other generators?
      TODO: Why no half?
      -> see normal distribution's todo
      - ✅
      - ✅
    *
      - float4
      - Philox 4x32-10
      - ✅
      - ✅
    *
      - double4
      - Philox 4x32-10
      - ✅
      - ❌

Poisson distributions
"""""""""""""""""""""

 .. list-table:: Supported types for poisson distributions
    :header-rows: 1
    :name: device-types-poisson-distribution

    *
      - Type
      - rocRAND support
      - supported rocRAND generators
      - cuRAND support
    *
      - unsigned int
      - Philox 4x32-10, mrg31k3p, mrg32k3a, xorwow, mtgp32, sobol32, scrambled sobol32, lfsr113, all ThreeFry generators
      - ✅
      - ✅
    *
      - unsigned long long
      - sobol64, scrambled sobol64
      - ✅
      - ✅
    *
      - uint4
      - philox 4x32-10
      - ✅
      - ✅ philox 4x32-10

Discrete distributions
""""""""""""""""""""""

 .. list-table:: Supported types for discrete distributions
    :header-rows: 1
    :name: device-types-discrete-distribution

    *
      - Type
      - rocRAND support
      - supported rocRAND generators
      - cuRAND support
    *
      - unsigned int
      - all generators
      - ✅
      - ✅
    *
      - uint4
      - Philox 4x32-10
      - ✅
      - ✅ - only Philox - 4x32-10

.. rubric:: Footnotes
.. [#] Generation of 64 bit :code:`unsigned long long` integers is only supported by 64 bit generators (Scrambled Sobol 64, Sobol64, Threefry 2x64-20 and Threefry 4x64-20).