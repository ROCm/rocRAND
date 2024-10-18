.. meta::
  :description: rocRAND documentation and API reference library
  :keywords: rocRAND, ROCm, API, documentation

.. _rocrand-docs-home:

********************************************************************
rocRAND documentation
********************************************************************

rocRAND provides functions that generate pseudo-random and quasi-random numbers. The rocRAND library is implemented in the `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_
programming language and optimized for AMD's latest discrete GPUs. It is designed to run on top
of AMD's `ROCm <https://rocm.docs.amd.com/en/latest/>`_, but it also works on NVIDIA CUDA-enabled GPUs.

rocRAND includes a wrapper library called hipRAND, which you can use to easily port
NVIDIA CUDA applications using the CUDA cuRAND library to the
`HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`_ layer. In the
`ROCm <https://rocm.docs.amd.com/en/latest/>`_ environment, hipRAND uses rocRAND.

You can access rocRAND code on our `GitHub repository <https://github.com/ROCm/rocRAND>`_.

The documentation is structured as follows:

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :ref:`installing`

  .. grid-item-card:: Conceptual

    * :ref:`programmers-guide`

    * :ref:`curand-compatibility`
    * :ref:`dynamic-ordering-configuration`

  .. grid-item-card:: API reference

    * :doc:`rocRAND data type support <api-reference/data-type-support>`
    * :ref:`cpp-api`
    * :ref:`python-api`
    * :doc:`Fortran API reference <fortran-api-reference>`
    * :doc:`API library <doxygen/html/index>`

To contribute to the documentation, refer to
`Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.
