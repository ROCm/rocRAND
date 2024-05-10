.. meta::
  :description: rocRAND documentation and API reference library
  :keywords: rocRAND, ROCm, API, documentation
  
.. _python-api:

====================
Python API reference
====================

This chapter describes the rocRAND Python module API.

.. default-domain:: py
.. py:currentmodule:: rocrand

class PRNG
----------

.. autoclass:: rocrand.PRNG
   :inherited-members:
   :members:

class QRNG
----------

.. autoclass:: rocrand.QRNG
   :inherited-members:
   :members:

Exceptions
----------

.. autoexception:: rocrand.RocRandError
   :members:

.. autoexception:: rocrand.HipError
   :members:

Utilities
---------

.. autoclass:: rocrand.DeviceNDArray
   :members:

.. autofunction:: rocrand.empty

.. autofunction:: rocrand.get_version
