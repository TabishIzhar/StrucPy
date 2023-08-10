# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../../StrucPy'))       # ../../ two directories up RCFA.py

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'StrucPy'
copyright = '2023, Tabish Izhar'
author = 'Tabish Izhar'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autosectionlabel',
              'autoapi.extension']        #'sphinx.ext.autodoc',

autosectionlabel_prefix_document = True
templates_path = ['_templates']
exclude_patterns = []
autoapi_dirs = ['../../StrucPy']
autoapi_options =  [ 'members', 'show-inheritance', 'show-module-summary', 'special-members', 'imported-members', ]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
