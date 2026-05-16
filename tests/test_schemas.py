"""Schema and dataclass serialization tests for contract artifacts.

Confirms packaged schemas load correctly and dataclasses serialize cleanly.
"""

import json
from dataclasses import asdict

import pytest

import design_research_agents
from design_research_agents._contracts import (
    ExecutionResult,
    LLMResponse,
    ToolCostHints,
    ToolResult,
    ToolSpec,
)
from design_research_agents._schemas import SCHEMA_NAMES, load_schema
from design_research_agents._schemas._validation import (
    SchemaValidationError,
    validate_payload_against_schema,
)


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
    agent_result = ExecutionResult(
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
    assert hasattr(design_research_agents, "Toolbox")
    assert "Toolbox" in design_research_agents.__all__
    assert "tools" not in design_research_agents.__all__
    assert not hasattr(design_research_agents, "BaseToolRuntime")
    from design_research_agents import tools as tools_module

    assert not hasattr(tools_module, "BaseToolRuntime")


def test_schema_validation_supports_anyof_enum_and_arrays() -> None:
    validate_payload_against_schema(
        payload={"a": 1},
        schema={
            "anyOf": [
                {"type": "object", "required": ["a"]},
                {"type": "object", "required": ["b"]},
            ]
        },
    )

    validate_payload_against_schema(
        payload="x",
        schema={"enum": ["x", "y"]},
    )

    with pytest.raises(SchemaValidationError, match="not in enum"):
        validate_payload_against_schema(payload="z", schema={"enum": ["x", "y"]})

    validate_payload_against_schema(
        payload=[{"n": 1}, {"n": 2}],
        schema={
            "type": "array",
            "items": {
                "type": "object",
                "required": ["n"],
                "properties": {"n": {"type": "integer"}},
                "additionalProperties": False,
            },
        },
    )

    with pytest.raises(SchemaValidationError, match="unexpected field"):
        validate_payload_against_schema(
            payload={"a": 1, "extra": 2},
            schema={
                "type": "object",
                "required": ["a"],
                "properties": {"a": {"type": "integer"}},
                "additionalProperties": False,
            },
        )


def test_schema_validation_type_union_and_null_paths() -> None:
    validate_payload_against_schema(payload=3, schema={"type": ["string", "integer"]})
    validate_payload_against_schema(payload=None, schema={"type": "null"})
    validate_payload_against_schema(payload=True, schema={"type": "boolean"})
    validate_payload_against_schema(payload=1.2, schema={"type": "number"})

    with pytest.raises(SchemaValidationError, match="type mismatch"):
        validate_payload_against_schema(payload=1.2, schema={"type": ["string", "object"]})

    with pytest.raises(SchemaValidationError, match="expected integer"):
        validate_payload_against_schema(payload=True, schema={"type": "integer"})

    with pytest.raises(SchemaValidationError, match="expected null"):
        validate_payload_against_schema(payload="x", schema={"type": "null"})

    with pytest.raises(SchemaValidationError, match="invalid JSON schema"):
        validate_payload_against_schema(payload="ok", schema={"type": ["string", 1]})
