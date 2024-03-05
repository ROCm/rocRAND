.. meta::
   :description: rocRAND provides functions that generate pseudo-random and quasi-random numbers
   :keywords: rocRAND, ROCm, library, API, HIP

.. _what-is-rocrand:

==================
What is rocRAND?
==================

rocRAND provides functions that generate pseudo-random and quasi-random numbers.

The rocRAND library is implemented in the `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_
programming language and optimized for AMD's latest discrete GPUs. It is designed to run on top
of AMD's `ROCm <https://rocm.docs.amd.com/en/latest/>`_, but it also works on CUDA-enabled GPUs.

rocRAND includes a wrapper library called hipRAND, which you can use to easily port
CUDA applications using the cuRAND library to the
`HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ layer. In the
`ROCm <https://rocm.docs.amd.com/en/latest/>`_ environment, hipRAND uses rocRAND.
