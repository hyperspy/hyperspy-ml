# -*- coding: utf-8 -*-
import os
import sys
from datetime import datetime

import hyperspy_ml

sys.path.insert(0, os.path.abspath("../"))

# -- General configuration -----------------------------------------------

extensions = [
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
source_suffix = ".rst"
master_doc = "index"

project = "HyperSpy ML"
copyright = f"2024-{datetime.today().year}, The HyperSpy development team"
author = "The HyperSpy development team"

release = hyperspy_ml.__version__
version = ".".join(release.split(".")[:2])

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

# -- Options for HTML output ---------------------------------------------

html_theme = "pydata_sphinx_theme"
html_logo = "_static/hyperspy_logo.png"
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/hyperspy/hyperspy-ml",
    "logo": {
        "text": "HyperSpy ML",
    },
    "show_version_warning_banner": True,
}

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "hyperspy": ("https://hyperspy.org/hyperspy-doc/current", None),
}

# -- Options for numpydoc extension -----------------------------------

numpydoc_show_class_members = False
autodoc_default_options = {
    "show-inheritance": True,
}
