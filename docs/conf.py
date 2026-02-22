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
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.mermaid",
]

# Docstring style: prefer Google-style (works well with type hints).
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = False

# Keep type hints out of rendered docs to avoid unresolved nitpicky targets.
autodoc_typehints = "none"

# Generate autosummary stub pages at build time.
autosummary_generate = True
autosummary_imported_members = True

# Treat unresolved cross references as errors.
nitpicky = True
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
nitpick_ignore = [
    ("py:class", "ExecutionResult"),
    ("py:class", "LlamaCppServerBackend"),
    ("py:class", "design_research_agents._contracts._agent.ExecutionResult"),
    ("py:class", "design_research_agents._contracts._llm.LLMResponse"),
    ("py:class", "design_research_agents._contracts._tools.ToolCostHints"),
    ("py:class", "design_research_agents._contracts._tools.ToolMetadata"),
    ("py:class", "design_research_agents._contracts._tools.ToolResult"),
    ("py:class", "design_research_agents._model_selection._catalog.ModelCatalog"),
    ("py:class", "design_research_agents._model_selection._types.ModelCostHint"),
    ("py:class", "design_research_agents._model_selection._types.ModelLatencyHint"),
    ("py:class", "design_research_agents._model_selection._types.ModelMemoryHint"),
    ("py:class", "design_research_agents._model_selection._types.ModelSafetyConstraints"),
    (
        "py:class",
        "design_research_agents._model_selection._types.ModelSelectionPolicyConfig",
    ),
    ("py:class", "design_research_agents._model_selection._types.ModelSpec"),
    ("py:exc", "SchemaValidationError"),
]

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

html_logo = "drc.png"
html_theme_options = {
    "logo_only": True,
}

# Linkcheck tuning for stable CI behavior.
linkcheck_retries = 2
linkcheck_timeout = 10
linkcheck_workers = 10
linkcheck_anchors = False
linkcheck_ignore = [
    r"https://api\.example\.com/.*",
]


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
