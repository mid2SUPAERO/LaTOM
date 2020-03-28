# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'LaTOM'
copyright = '2020, Alberto Fossa\', Giuliana Elena Miceli'
author = 'Alberto Fossa\', Giuliana Elena Miceli'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autosummary',
              'sphinx.ext.autodoc',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.graphviz',
              'sphinx.ext.inheritance_diagram',
              'sphinx_autopackagesummary',
              'sphinx_rtd_theme',
              'recommonmark',
              'sphinx_markdown_tables',
              'sphinx_gallery.gen_gallery'
]

# Numpy settings
numpydoc_show_class_members = False

# Autosummary settings
automodsumm_inherited_members = True
autosummary_generate = True
autosummary_imported_members = True

# Autodoc settings
autodoc_inherit_docstrings = True
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'inherited-members': True,
    'show-inheritance': True,
    'undoc-members': True,
    'exclude-members': '__str__'
}

source_suffix = ['.rst', '.md']

# Napoleon settings
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# Sphinx Gallery settings
sphinx_gallery_conf = {
    'examples_dirs': '../../scripts',
    'gallery_dirs': 'examples',
    'default_thumb_file': 'source/example_traj.png',
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = 'sphinx_rtd_theme'
html_show_sourcelink = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# 'sphinx_rtd_theme' options
html_logo = 'LaTOM_logo.png'
