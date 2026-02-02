import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../../src"))

project = "splade-repro"
author = "splade-repro"
copyright = f"{datetime.now().year}, {author}"

extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon"]
templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
