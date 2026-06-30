# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
sys.path.insert(0, os.path.abspath('../../src')) # Adjust path to point to your app root

project = 'Netmap'
copyright = '2025, Anne Hartebrodt, Mhaned Oubounyt'
author = 'Anne Hartebrodt, Mhaned Oubounyt'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx_autodoc_typehints',
    'nbsphinx',
]

# Never re-execute notebooks during the Sphinx build.
# Notebooks must be pre-run (outputs saved) before committing.
nbsphinx_execute = 'never'

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs', None),
    'anndata': ('https://anndata.readthedocs.io/en/stable', None),
    'scanpy': ('https://scanpy.readthedocs.io/en/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_attr_annotations = False

autodoc_default_options = {
    'undoc-members': False,
    'show-inheritance': True,
}

autodoc_mock_imports = [
    'numpy',
    'pandas',
    'torch',
    'anndata',
    'scanpy',
    'captum',
    'pyarrow',
    'h5py',
    'pyvis',
    'networkx',
    'sklearn',
    'scipy',
    'statsmodels',
    'seaborn',
    'matplotlib',
    'tqdm',
    'pyucell',
    'requests',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = []

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True}

# This line is crucial for the :recursive: option to work!
autosummary_generate = True