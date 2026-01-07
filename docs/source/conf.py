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
    'sphinx.ext.napoleon', # Highly recommended for Google/NumPy style docstrings
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True}

# This line is crucial for the :recursive: option to work!
autosummary_generate = True