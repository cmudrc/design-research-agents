"""Schema-focused tests for packaged JSON schemas and contract dataclasses."""

import json
from dataclasses import asdict

from design_research_agents.contracts import (
    AgentResult,
    LLMResponse,
    ToolCostHints,
    ToolResult,
    ToolSpec,
)
from design_research_agents.schemas import SCHEMA_NAMES, SCHEMA_VERSION, load_schema


def test_all_schemas_load_from_packaged_resources() -> None:
    # Verify every shipped schema is discoverable and versioned consistently.
    for name in SCHEMA_NAMES:
        schema = load_schema(name)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        schema_id = str(schema["$id"])
        assert f"/schemas/{SCHEMA_VERSION}/" in schema_id
        assert schema_id.endswith(f"{name}.schema.json")


def test_tool_spec_serializes_and_deserializes_cleanly() -> None:
    # ToolSpec should round-trip through JSON without losing typed fields.
    tool_spec = ToolSpec(
        name="calculator_tool",
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
    assert round_trip["name"] == "calculator_tool"
    assert round_trip["permissions"] == ["compute:arithmetic"]
    assert round_trip["cost_hints"]["token_cost_estimate"] == 3


def test_tool_result_and_agent_result_serialize_and_deserialize_cleanly() -> None:
    # Result dataclasses should serialize cleanly for logging and persistence.
    tool_result = ToolResult(
        tool_name="calculator_tool",
        output={"expression": "6*7", "result": 42},
        success=True,
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
    assert round_trip_tool_result["tool_name"] == "calculator_tool"
    assert round_trip_tool_result["success"] is True

    serialized_agent_result = json.dumps(asdict(agent_result))
    round_trip_agent_result = json.loads(serialized_agent_result)
    assert round_trip_agent_result["output"]["final"] == "hello"
    assert round_trip_agent_result["tool_results"][0]["tool_name"] == "calculator_tool"
    assert round_trip_agent_result["model_response"]["model"] == "base-model"
