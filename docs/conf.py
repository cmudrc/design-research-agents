"""Sphinx configuration for the project documentation."""

import os
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
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]
if os.environ.get("DRA_DOCS_ENABLE_INTERSPHINX") == "1":
    extensions.append("sphinx.ext.intersphinx")

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
intersphinx_mapping = (
    {
        "python": ("https://docs.python.org/3", None),
    }
    if "sphinx.ext.intersphinx" in extensions
    else {}
)
nitpick_ignore = [
    ("py:class", "ExecutionResult"),
    ("py:class", "LlamaCppServerBackend"),
    ("py:class", "design_research_agents._contracts._delegate.Delegate"),
    ("py:class", "design_research_agents._contracts._delegate.ExecutionResult"),
    ("py:class", "design_research_agents._contracts._execution.ExecutionResult"),
    ("py:class", "design_research_agents._contracts._llm.LLMResponse"),
    ("py:class", "design_research_agents._contracts._llm.LLMRequest"),
    ("py:class", "design_research_agents._contracts._memory.MemoryWriteRecord"),
    ("py:class", "design_research_agents._contracts._tools.ToolCostHints"),
    ("py:class", "design_research_agents._contracts._tools.ToolMetadata"),
    ("py:class", "design_research_agents._contracts._tools.ToolResult"),
    ("py:class", "design_research_agents._contracts._workflow.DelegateBatchCall"),
    ("py:class", "design_research_agents._contracts._workflow.DelegateRunner"),
    ("py:class", "design_research_agents._contracts._workflow.WorkflowArtifact"),
    ("py:class", "design_research_agents._contracts._workflow.WorkflowDelegate"),
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

# HTML output theme.
# Prefer the Read the Docs theme, with a local fallback when it is unavailable.
if os.environ.get("READTHEDOCS") == "True":
    html_theme = "sphinx_rtd_theme"
else:
    try:
        import sphinx_rtd_theme  # noqa: F401

        html_theme = "sphinx_rtd_theme"
    except ImportError:
        html_theme = "alabaster"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = "drc.png"
html_favicon = "_static/favicon.ico"
html_title = project
html_theme_options = {
    "logo_only": False,
}

# Linkcheck tuning for stable CI behavior.
linkcheck_retries = 2
linkcheck_timeout = 10
linkcheck_workers = 10
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
