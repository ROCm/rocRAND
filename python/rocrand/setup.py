from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

setup(
    name="rocrand",
    version="1.6.0",
    description="rocRAND Python Wrapper",
    long_description=readme,
    author="Advanced Micro Devices, Inc.",
    # author_email="",
    url="https://github.com/ROCmSoftwarePlatform/rocRAND",
    license="MIT",
    packages=["rocrand"],
    install_requires=["numpy"],
    test_suite="tests"
)
