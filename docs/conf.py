# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import shutil
import sys

from rocm_docs import ROCmDocs

# We need to add the location of the rocrand Python module to the PATH
# in order to build the documentation of that module
docs_dir_path = pathlib.Path(__file__).parent
python_dir_path = docs_dir_path.parent / 'python' / 'rocrand'
sys.path.append(str(python_dir_path))

shutil.copy2('../library/src/fortran/README.md', './fortran-README.md')

external_projects_current_project = "rocrand"

docs_core = ROCmDocs("rocRAND Documentation")
docs_core.run_doxygen(doxygen_root=".doxygen", doxygen_path=".doxygen/xml")
docs_core.setup()

for sphinx_var in ROCmDocs.SPHINX_VARS:
    globals()[sphinx_var] = getattr(docs_core, sphinx_var)
