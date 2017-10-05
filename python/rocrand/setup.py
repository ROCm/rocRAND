from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

version = "1.0"
release = "1.0.0"
setup(
    name="rocrand",
    version=release,
    description="rocRAND Python Wrapper",
    long_description=readme,
    author="Advanced Micro Devices, Inc.",
    # author_email="",
    url="https://github.com/ROCmSoftwarePlatform/rocRAND",
    license="MIT",
    packages=["rocrand"],
    install_requires=["numpy"],
    test_suite="tests",
    command_options={
        "build_sphinx": {
            "version": ("setup.py", version),
            "release": ("setup.py", release)}},
)
