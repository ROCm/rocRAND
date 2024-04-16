.. meta::
   :description: rocRAND provides functions that generate pseudo-random and quasi-random numbers
   :keywords: rocRAND, ROCm, library, API, HIP

.. _what-is-rocrand:

====================================
What is rocRAND?
====================================

rocRAND provides functions that generate pseudo-random and quasi-random numbers.

The rocRAND library is implemented in the :doc:`HIP <hip:index>` programming language and
optimized for AMD's latest discrete GPUs. It is designed to run on top of AMD's
:doc:`ROCm <rocm:index>` runtime, but it also works on CUDA-enabled GPUs. Additionally, the project
includes a wrapper library called hipRAND which allows users to easily port CUDA applications that use
cuRAND library to the :doc:`HIP <hip:index>` layer. In a :doc:`ROCm <rocm:index>` environment,
hipRAND uses rocRAND, however in CUDA environment cuRAND is used instead.
