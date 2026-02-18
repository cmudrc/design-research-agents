from __future__ import annotations

import json

from design_research_agents.agent.internal.response_schemas import (
    build_router_selection_response_schema,
)
from design_research_agents.agent.internal.router_agent_helpers import (
    ToolAlternative,
    parse_route_response,
    resolve_model_route,
)


def test_router_selection_schema_requires_tool_names_only() -> None:
    schema = build_router_selection_response_schema(alternative_identifiers=["alpha", "beta"])

    assert schema["required"] == ["tool_names"]
    assert "selection" not in schema["properties"]
    assert schema["properties"]["tool_names"]["items"]["enum"] == ["alpha", "beta"]
    assert schema["properties"]["tool_names"]["minItems"] == 1


def test_parse_route_response_rejects_legacy_aliases() -> None:
    assert parse_route_response(json.dumps({"selection": 0})) is None
    assert parse_route_response(json.dumps({"selected_alternative_index": 1})) is None
    assert parse_route_response(json.dumps({"tool_name": "alpha"})) is None
    assert parse_route_response(json.dumps({"name": "alpha"})) is None


def test_parse_route_response_accepts_tool_names_and_resolves_first_match() -> None:
    parsed = parse_route_response(
        json.dumps({"tool_names": ["beta", "beta", "alpha"], "reason": "best fit"})
    )
    assert parsed is not None
    assert parsed.tool_names == ("beta", "alpha")
    assert parsed.reason == "best fit"

    resolved = resolve_model_route(
        parsed_route=parsed,
        alternatives=[
            ToolAlternative(
                tool_name="alpha",
                description="alpha",
                input_schema={"type": "object"},
            ),
            ToolAlternative(
                tool_name="beta",
                description="beta",
                input_schema={"type": "object"},
            ),
        ],
    )
    assert resolved is not None
    selected_alternative, selected_index, reason, selected_names = resolved
    assert selected_alternative.tool_name == "beta"
    assert selected_index == 1
    assert reason == "best fit"
    assert selected_names == ["beta", "alpha"]


def test_resolve_model_route_returns_none_for_unknown_tool_name() -> None:
    parsed = parse_route_response(json.dumps({"tool_names": ["unknown"]}))
    assert parsed is not None
    resolved = resolve_model_route(
        parsed_route=parsed,
        alternatives=[
            ToolAlternative(
                tool_name="alpha",
                description="alpha",
                input_schema={"type": "object"},
            )
        ],
    )
    assert resolved is None
