"""Loader and renderer for packaged prompt templates.

Prompt files live in package resources so they can be versioned independently
from agent code and loaded consistently across environments.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from importlib.resources import files
from string import Template
from typing import Final

_PROMPT_FILES: Final[dict[str, str]] = {
    "router_system": "router_system.md",
    "router_user_route": "router_user_route.md",
    "tool_calling_system": "tool_calling_system.md",
    "tool_calling_user_select_tool": "tool_calling_user_select_tool.md",
    "single_step_code_system": "single_step_code_system.md",
    "single_step_code_user_plan": "single_step_code_user_plan.md",
    "multi_step_continue_system": "multi_step_continue_system.md",
    "multi_step_continue_user": "multi_step_continue_user.md",
    "multi_step_step_user": "multi_step_step_user.md",
    "multi_step_json_step_user": "multi_step_json_step_user.md",
}

PROMPT_NAMES: Final[tuple[str, ...]] = tuple(sorted(_PROMPT_FILES))


@cache
def load_prompt(name: str) -> str:
    """Load a prompt template by logical name from packaged resources.

    The function validates that the name is registered and that the resolved
    file is non-empty, then returns a stripped text template ready for rendering.

    Args:
        name: Logical prompt template name in the registry.

    Returns:
        The stripped prompt template text.

    Raises:
        ValueError: If the prompt name is unknown or the template is empty.
    """
    filename = _PROMPT_FILES.get(name)
    if filename is None:
        raise ValueError(f"Unknown prompt template '{name}'.")

    resource = files("design_research_agents.prompts").joinpath(filename)
    with resource.open("r", encoding="utf-8") as prompt_file:
        content = prompt_file.read().strip()
    if not content:
        raise ValueError(f"Prompt template '{name}' is empty.")
    return content


def render_prompt(name: str, *, variables: Mapping[str, object]) -> str:
    """Render a prompt template with ``string.Template`` variable substitution.

    All variable values are coerced to strings before substitution. Missing
    variables raise ``ValueError`` with the missing key name for easier debugging.

    Args:
        name: Logical prompt template name in the registry.
        variables: Mapping of template variable names to replacement values.

    Returns:
        The rendered prompt text.

    Raises:
        ValueError: If the prompt name is unknown or a variable is missing.
    """
    normalized_variables = {key: str(value) for key, value in variables.items()}
    template = Template(load_prompt(name))
    try:
        return template.substitute(normalized_variables)
    except KeyError as exc:
        missing_key = exc.args[0] if exc.args else "unknown"
        raise ValueError(
            f"Prompt template '{name}' is missing required variable '{missing_key}'."
        ) from exc
