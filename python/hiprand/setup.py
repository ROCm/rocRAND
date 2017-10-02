from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

setup(
    name="hiprand",
    version="1.6.0",
    description="hipRAND Python Wrapper",
    long_description=readme,
    author="Advanced Micro Devices, Inc.",
    # author_email="",
    url="https://github.com/ROCmSoftwarePlatform/rocRAND",
    license="MIT",
    packages=["hiprand"],
    install_requires=["numpy"],
    test_suite="tests"
)
