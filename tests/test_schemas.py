"""Schema and dataclass serialization tests for contract artifacts.

Confirms packaged schemas load correctly and dataclasses serialize cleanly.
"""

import json
from dataclasses import asdict

import pytest

import design_research_agents
from design_research_agents.contracts import (
    AgentResult,
    LLMResponse,
    ToolCostHints,
    ToolResult,
    ToolSpec,
)
from design_research_agents.schemas import SCHEMA_NAMES, load_schema


def test_all_schemas_load_from_packaged_resources() -> None:
    # Verify every shipped schema is discoverable and consistently identified.
    for name in SCHEMA_NAMES:
        schema = load_schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        schema_id = str(schema["$id"])
        assert "/schemas/" in schema_id
        assert "/schemas/v" not in schema_id
        assert schema_id.endswith(f"{name}.schema.json")


def test_tool_spec_serializes_and_deserializes_cleanly() -> None:
    # ToolSpec should round-trip through JSON without losing typed fields.
    tool_spec = ToolSpec(
        name="calculator",
        description="Calculator tool",
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        permissions=("compute:arithmetic",),
        cost_hints=ToolCostHints(
            token_cost_estimate=3,
            latency_ms_estimate=2,
            usd_cost_estimate=0.0,
        ),
    )
    serialized = json.dumps(asdict(tool_spec))
    round_trip = json.loads(serialized)
    assert round_trip["name"] == "calculator"
    assert round_trip["metadata"]["source"] == "core"
    assert round_trip["permissions"] == ["compute:arithmetic"]
    assert round_trip["cost_hints"]["token_cost_estimate"] == 3


def test_tool_result_and_agent_result_serialize_and_deserialize_cleanly() -> None:
    # Result dataclasses should serialize cleanly for logging and persistence.
    tool_result = ToolResult(
        tool_name="calculator",
        ok=True,
        result={"expression": "6*7", "result": 42},
        metadata={"source": "unit-test"},
    )
    agent_result = AgentResult(
        output={"final": "hello"},
        success=True,
        tool_results=[tool_result],
        model_response=LLMResponse(
            model="base-model",
            text='{"final":"hello"}',
            raw_output={"final": "hello"},
        ),
        metadata={"trace_id": "abc"},
    )

    serialized_tool_result = json.dumps(asdict(tool_result))
    round_trip_tool_result = json.loads(serialized_tool_result)
    assert round_trip_tool_result["tool_name"] == "calculator"
    assert round_trip_tool_result["ok"] is True
    assert round_trip_tool_result["result"]["result"] == 42

    serialized_agent_result = json.dumps(asdict(agent_result))
    round_trip_agent_result = json.loads(serialized_agent_result)
    assert round_trip_agent_result["output"]["final"] == "hello"
    assert round_trip_agent_result["tool_results"][0]["tool_name"] == "calculator"
    assert round_trip_agent_result["model_response"]["model"] == "base-model"


def test_tool_result_rejects_legacy_success_and_output_fields() -> None:
    with pytest.raises(TypeError):
        ToolResult(
            tool_name="calculator",
            success=True,  # type: ignore[call-arg]
            output={"result": 42},  # type: ignore[call-arg]
        )


def test_package_no_longer_exports_base_tool_runtime() -> None:
    assert hasattr(design_research_agents, "tools")
    assert hasattr(design_research_agents.tools, "UnifiedToolRuntime")
    assert not hasattr(design_research_agents, "UnifiedToolRuntime")
    assert not hasattr(design_research_agents, "BaseToolRuntime")
    assert not hasattr(design_research_agents.tools, "BaseToolRuntime")
