# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import re
import sys

from rocm_docs import ROCmDocs

# We need to add the location of the rocrand Python module to the PATH
# in order to build the documentation of that module
docs_dir_path = pathlib.Path(__file__).parent
python_dir_path = docs_dir_path.parent / 'python' / 'rocrand'
sys.path.append(str(python_dir_path))

with open('../CMakeLists.txt', encoding='utf-8') as f:
    match = re.search(r'rocm_setup_version\( VERSION\s+\"?([0-9.]+)[^0-9.]+', f.read())
    if not match:
        raise ValueError("VERSION not found!")
    version_number = match[1]
left_nav_title = f"rocRAND {version_number} Documentation"

# for PDF output on Read the Docs
project = "rocRAND Documentation"
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved."
version = version_number
release = version_number

external_toc_path = "./sphinx/_toc.yml"

docs_core = ROCmDocs(left_nav_title)
docs_core.run_doxygen(doxygen_root="doxygen", doxygen_path="doxygen/xml")
docs_core.setup()

external_projects_current_project = "rocrand"

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
