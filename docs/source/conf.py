# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'LHD-FICXS-py'
copyright = '2025, W.H.J. Hayashi'
author = 'W.H.J. Hayashi'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

autodoc_mock_imports = [
    "f90nml",
    "h5py",
    "matplotlib",
    "netCDF4",
    "numpy",
    "numpydoc",
    "pandas",
    "pyarrow",
    "scikit-image",
    "scipy",
    "sphinx",
]
