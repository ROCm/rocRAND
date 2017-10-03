Installation
============

Requirements
------------

* rocRAND
* Python 2.7 or 3.5
* NumPy (will be installed automatically as a dependency if necessary)

Note: If rocRAND is built from sources but not installed or installed in
non-standard directory set ``ROCRAND_PATH`` environment variable, for example::

    export ROCRAND_PATH=~/rocRAND/build/library/


Installing
----------

Install::

    cd python/rocrand
    pip install .

Run tests::

    python tests/rocrand_test.py
