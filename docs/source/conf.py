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
#import sphinx_fontawesome
import sphinx_rtd_theme

# -- Project information -----------------------------------------------------

project = 'SMaLL best practice'
copyright = u'\
2021, Multi-Scale ModeLing Laboratory (SMaLL) \
POLITECNICO DI TORINO (Department of Energy "Galileo Ferraris" (DENERG)), ITA . All rights reserved'
author = 'SMaLL Team'


# The full version, including alpha/beta/rc tags
release = 'latest'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc', 'sphinx.ext.doctest', 'sphinx.ext.todo',
    'sphinx.ext.ifconfig', 'sphinx.ext.intersphinx', 'sphinx.ext.viewcode',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive', 'myst_nb', 'sphinx.ext.extlinks',
    'sphinx.ext.mathjax', 'sphinx_copybutton', 'sphinx_panels', 'sphinx_tabs.tabs',
    'sphinx_rtd_theme',
    #'sphinx_fontawesome'
]

ipython_mplbackend = ""

panels_add_bootstrap_css = True

copybutton_selector = 'div:not(.no-copy)>div.highlight pre'
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
#copybutton_prompt_text = '>>> |\\\\$ |In \\\\[\\\\d+\\\\]: |\\\\s+\\.\\.\\.: '
copybutton_prompt_is_regexp = True

sphinx_tabs_valid_builders = ['linkcheck']

todo_include_todos = True

extlinks = {
    'doi': ('https://doi.org/%s', 'doi:'),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The master toctree document.
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
#html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_path = ["_themes"]
html_logo = "logo.svg"
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
}
# Enable labeling for figures
numfig = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# -- Options for LaTeX output --------------------------------------------------

latex_engine = 'xelatex'

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    'preamble':
    r'''
\usepackage{amsmath,amsfonts,amssymb,amsthm}
\usepackage{braket}
\usepackage{newunicodechar}
\newunicodechar{⏹}{\ensuremath{\square}}
''',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, documentclass [howto/manual]).
latex_documents = [
    ('index', 'aiida-tutorials.tex', u'AiiDA Tutorials',
     author.replace(',', r'\and'), 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# If true, show page references after internal links.
#latex_show_pagerefs = False

# If true, show URL addresses after external links.
#latex_show_urls = False

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
#latex_domain_indices = True
