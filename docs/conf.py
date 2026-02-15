"""Sphinx configuration.

This file configures how Sphinx builds the docs in `docs/`.
"""

import os
import re
import sys
from pathlib import Path

from sphinx.application import Sphinx

# Include class docs and __init__
autoclass_content = "both"

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

# HTML output theme.
# Prefer the Read the Docs theme; keep a local fallback when not installed.
if os.environ.get("READTHEDOCS") == "True":
    html_theme = "sphinx_rtd_theme"
else:
    try:
        import sphinx_rtd_theme  # noqa: F401

        html_theme = "sphinx_rtd_theme"
    except ImportError:
        html_theme = "alabaster"

html_static_path = ["_static"]


_VIEWPORT_META_RE = re.compile(r'<meta name="viewport"[^>]*>', re.IGNORECASE)


def _dedupe_viewport_meta(
    app: object,
    pagename: str,
    templatename: str,
    context: dict[str, object],
    doctree: object,
) -> None:
    """Keep one viewport tag by removing extra entries from Sphinx metatags.

    Args:
        app: Sphinx application instance.
        pagename: Current page name.
        templatename: Template name for the page.
        context: Template context mapping.
        doctree: Document tree for the page.
    """
    del app, pagename, templatename, doctree
    metatags = context.get("metatags")
    if isinstance(metatags, str):
        context["metatags"] = _VIEWPORT_META_RE.sub("", metatags)


def setup(app: Sphinx) -> None:
    """Register build-time hooks.

    Args:
        app: Sphinx application instance.
    """
    app.connect("html-page-context", _dedupe_viewport_meta)
