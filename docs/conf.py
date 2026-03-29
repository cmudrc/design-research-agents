"""Sphinx configuration for the project documentation."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from sphinx.application import Sphinx

# Include class docs and ``__init__`` docstrings.
autoclass_content = "both"

# Add ``src/`` so autodoc imports the in-workspace package build.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Project metadata shown in generated documentation.
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

# Keep type hints out of rendered docs so nitpicky mode stays focused on explicit refs.
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
    ("py:class", "design_research_agents._contracts._delegate.ExecutionResult"),
    ("py:class", "design_research_agents._contracts._execution.ExecutionResult"),
    ("py:class", "design_research_agents._contracts._llm.LLMResponse"),
    ("py:class", "design_research_agents._contracts._llm.LLMRequest"),
    ("py:class", "design_research_agents._contracts._memory.MemoryWriteRecord"),
    ("py:class", "design_research_agents._contracts._tools.ToolCostHints"),
    ("py:class", "design_research_agents._contracts._tools.ToolMetadata"),
    ("py:class", "design_research_agents._contracts._tools.ToolResult"),
    ("py:class", "design_research_agents._contracts._workflow.WorkflowArtifact"),
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
nitpick_ignore_regex = [
    # Offline docs builds do not resolve stdlib inventory targets via intersphinx.
    ("py:class", r"collections\.abc\..+"),
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

try:
    import pydata_sphinx_theme  # noqa: F401
except ImportError:
    html_theme = "alabaster"
    html_theme_options: dict[str, object] = {}
else:
    html_theme = "pydata_sphinx_theme"
    html_theme_options = {
        "logo": {
            "text": project,
            "image_light": "_static/drc-light.png",
            "image_dark": "_static/drc-dark.png",
        },
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/cmudrc/design-research-agents",
                "icon": "fa-brands fa-github",
            },
        ],
        "navbar_align": "content",
        "header_links_before_dropdown": 4,
        "show_nav_level": 2,
        "navigation_with_keys": True,
        "show_prev_next": False,
        "secondary_sidebar_items": ["page-toc"],
    }

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "_static/drc-light.png"
html_favicon = "_static/favicon.ico"
html_title = project
html_sidebars = (
    {
        "index": [],
        "examples/index": [],
    }
    if html_theme == "pydata_sphinx_theme"
    else {}
)

# Linkcheck tuning for stable CI behavior.
# Keep concurrency conservative so external docs hosts are less likely to rate-limit CI runners.
linkcheck_retries = 3
linkcheck_timeout = 20
linkcheck_workers = 5
linkcheck_anchors = False
linkcheck_ignore = [
    r"https://api\.example\.com/.*",
    # OpenAI docs intermittently return 403 to CI linkcheck user-agents.
    r"https://platform\.openai\.com/docs/.*",
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
