===================
C/C++ API Reference
===================

This chapter describes the rocRAND C and C++ API.

Overview
========

Generator types
---------------

There are two main classes of generator in rocRAND: Pseudo-Random Number Generators (PRNGs), and Quasi-Random Number Generators (QRNGs). The following pseudo-random number generators are available:

* XORWOW.
* MRG32K3A.
* MTGP32.
* Philox 4x32-10.
* MRG31K3P.
* LFSR113.
* MT19937.
* ThreeFry 2x32-20, 4x32-30, 2x64-20 and 4x64-20.

Additionally, the following quasi-random number generators are available:

* Sobol32.
* Sobol64.
* Scrambled Sobol32.
* Scrambled Sobol64.

Device Functions
================
.. doxygengroup:: rocranddevice

C Host API
==========
.. doxygengroup:: rocrandhost

C++ Host API Wrapper
====================
.. doxygengroup:: rocrandhostcpp
