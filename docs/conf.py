"""Sphinx configuration."""

# This file configures how Sphinx builds the docs in `docs/`.

import sys
from pathlib import Path

# Add the project `src/` directory to sys.path so autodoc can import the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Project metadata shown in the generated documentation.
project = "design-research-agents"
copyright = "2026, design-research-agents contributors"
author = "design-research-agents contributors"

# Sphinx extensions:
# - autodoc: pull docstrings from the code
# - napoleon: parse Google/NumPy-style docstrings
# - viewcode: add links to highlighted source
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# Docstring style: prefer Google-style (works well with type hints).
napoleon_google_docstring = True
napoleon_numpy_docstring = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output theme. (Alabaster is the Sphinx default theme.)
html_theme = "alabaster"
html_static_path = ["_static"]
