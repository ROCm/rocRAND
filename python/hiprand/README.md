# hipRAND Python Wrapper

## Requirements

* rocRAND
* Python 2.7 or 3.5
* NumPy (will be installed automatically as a dependency if necessary)

If rocRAND is built from sources but not installed or installed in non-standard
directory set `ROCRAND_PATH` environment variable, for example:

```
export ROCRAND_PATH=~/rocRAND/build/library/
```

## Installing

```
cd python/hiprand
pip install .
```

Run tests:

```
python tests/hiprand_test.py
```

It is also possible to test the wrapper without installing:

```
python setup.py test
```

## Building documentation

Install Sphinx (http://www.sphinx-doc.org/en/stable/index.html):

```
pip install Sphinx
```

Run:

```
cd python/hiprand
python setup.py build_sphinx
```

The documentation will be placed into `docs/build/html`.
