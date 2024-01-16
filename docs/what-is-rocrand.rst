.. meta::
   :description: rocRAND provides functions that generate pseudo-random and quasi-random numbers
   :keywords: rocRAND, ROCm, library, API, HIP

.. _what-is-rocrand:

==================
What is rocRAND?
==================

The rocRAND project provides functions that generate pseudo-random and quasi-random numbers.

The rocRAND library is implemented in the `HIP <https://github.com/ROCm-Developer-Tools/HIP>`_
programming language and optimised for AMD's latest discrete GPUs. It is designed to run on top
of AMD's Radeon Open Compute `ROCm <https://rocm.github.io/>`_ runtime, but it also works on
CUDA enabled GPUs.
Additionally, the project includes a wrapper library called hipRAND which allows users to easily port
CUDA applications that use cuRAND library to the `HIP <https://github.com/ROCm-Developer-Tools/HIP>`_
layer. In `ROCm <https://rocm.github.io/>`_ environment hipRAND uses rocRAND, however in CUDA
environment cuRAND is used instead.
