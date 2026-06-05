from __future__ import annotations

import pytest

from design_research_agents._contracts._tools import ToolSpec
from design_research_agents._implementations._shared._agent_internal._code_tool_agent_parsing import (
    AllowedTool,
    canonicalize_generated_code,
    compile_default_allowed_tools,
    extract_allowed_tools,
    extract_boolean,
    extract_positive_int,
    extract_python_code,
    match_fenced_code_block,
    normalize_allowed_tools,
)

pytestmark = pytest.mark.contract


def _runtime_specs() -> dict[str, ToolSpec]:
    return {
        "calculator": ToolSpec(
            name="calculator",
            description="Compute",
            input_schema={"type": "object", "additionalProperties": True},
            output_schema={"type": "object"},
        ),
        "text.word_count": ToolSpec(
            name="text.word_count",
            description="Count words",
            input_schema={"type": "object", "additionalProperties": True},
            output_schema={"type": "object"},
        ),
    }


def _allowed_tools() -> tuple[AllowedTool, ...]:
    return (
        AllowedTool(
            tool_name="calculator",
            description="Compute",
            input_schema={"type": "object"},
        ),
    )


def test_canonicalize_generated_code_rewrites_direct_tool_name_calls() -> None:
    raw_code = "\n".join(
        [
            "import calculator",
            "result = calculator({'expression': '2+2'})",
            "final_output = {'result': result.get('value')}",
        ]
    )

    normalized = canonicalize_generated_code(code_text=raw_code, allowed_tools=_allowed_tools())

    assert normalized.changed is True
    assert normalized.stripped_safe_tool_imports == 1
    assert normalized.rewritten_tool_calls == 1
    assert normalized.rewritten_direct_name_calls == 1
    assert "import calculator" not in normalized.code_text
    assert "call_tool(" in normalized.code_text


def test_canonicalize_generated_code_rewrites_module_attribute_style() -> None:
    raw_code = "calculator.calculator(expression='2+2')"

    normalized = canonicalize_generated_code(code_text=raw_code, allowed_tools=_allowed_tools())

    assert normalized.changed is True
    assert normalized.rewritten_module_attr_calls == 1
    assert normalized.rewritten_tool_calls == 1
    assert "call_tool(" in normalized.code_text


def test_canonicalize_generated_code_keeps_unsupported_call_patterns_unchanged() -> None:
    raw_code = "calculator(*values)"

    normalized = canonicalize_generated_code(code_text=raw_code, allowed_tools=_allowed_tools())

    assert normalized.changed is False
    assert normalized.rewritten_tool_calls == 0
    assert normalized.code_text == raw_code


def test_canonicalize_generated_code_surfaces_parse_errors_without_mutation() -> None:
    raw_code = "def bad(:\n  pass"

    normalized = canonicalize_generated_code(code_text=raw_code, allowed_tools=_allowed_tools())

    assert normalized.parse_error is not None
    assert normalized.code_text == raw_code
    assert normalized.raw_code_text == raw_code


def test_allowed_tool_normalization_and_default_compilation_are_runtime_bounded() -> None:
    specs = _runtime_specs()

    normalized = normalize_allowed_tools(
        raw_tools=[
            {"tool_name": "calculator", "description": "first"},
            {"name": "calculator", "description": "last-wins", "tool_input": {"x": 1}},
            {"tool_name": "unknown"},
            "bad-entry",
        ],
        runtime_specs=specs,
    )
    assert len(normalized) == 1
    assert normalized[0].tool_name == "calculator"
    assert normalized[0].description == "last-wins"
    assert normalized[0].default_tool_input == {"x": 1}

    compiled_defaults = compile_default_allowed_tools(runtime_specs=specs, default_tools=None)
    assert {tool.tool_name for tool in compiled_defaults} == {"calculator", "text.word_count"}


def test_allowed_tool_parsing_covers_invalid_entries_and_skill_activation_default() -> None:
    specs = {
        **_runtime_specs(),
        "skills.activate": ToolSpec(
            name="skills.activate",
            description="Activate a skill",
            input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
            output_schema={"type": "object"},
        ),
    }

    assert normalize_allowed_tools(raw_tools="calculator", runtime_specs=specs) == []
    assert normalize_allowed_tools(raw_tools=[{"tool_name": 3}, {"name": "   "}], runtime_specs=specs) == []
    normalized = normalize_allowed_tools(
        raw_tools=[
            {
                "tool_name": "calculator",
                "description": "custom",
                "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}}},
            }
        ],
        runtime_specs=specs,
    )
    assert normalized[0].input_schema["properties"] == {"expression": {"type": "string"}}

    compiled = compile_default_allowed_tools(
        runtime_specs=specs,
        default_tools=[{"tool_name": "calculator"}],
    )
    assert [tool.tool_name for tool in compiled] == ["calculator", "skills.activate"]

    cloned, source = extract_allowed_tools(default_allowed_tools=compiled)
    assert source == "init_default"
    assert cloned == list(compiled)
    assert cloned is not list(compiled)


def test_scalar_input_parsing_uses_defaults_for_invalid_values() -> None:
    payload = {"missing": None, "bool": True, "count": 3, "zero": 0, "flag": False, "bad_flag": "yes"}

    assert extract_positive_int(input_payload=payload, key="missing", default_value=9) == 9
    assert extract_positive_int(input_payload=payload, key="bool", default_value=9) == 9
    assert extract_positive_int(input_payload=payload, key="count", default_value=9) == 3
    assert extract_positive_int(input_payload=payload, key="zero", default_value=9) == 9
    assert extract_boolean(input_payload=payload, key="flag", default_value=True) is False
    assert extract_boolean(input_payload=payload, key="bad_flag", default_value=True) is True


def test_python_code_extraction_prefers_python_fences_only() -> None:
    assert match_fenced_code_block("```js\nx=1\n```") is None
    assert match_fenced_code_block("```python") is None
    assert match_fenced_code_block("```python\nx=1") is None
    assert match_fenced_code_block("```python\nx=1\n```") == "x=1\n"
    assert extract_python_code("```py\nx=2\n``` trailing") == "x=2"
    assert extract_python_code(" print('x') ") == "print('x')"


def test_canonicalize_generated_code_covers_empty_and_unrewritable_call_shapes() -> None:
    allowed_tools = _allowed_tools()

    empty = canonicalize_generated_code(code_text="", allowed_tools=allowed_tools)
    assert empty.code_text == ""
    assert empty.metadata()["changed"] is False

    for raw_code in (
        "calculator(1, 2)",
        "calculator()",
        "calculator(**payload)",
        "unknown({'x': 1})",
        "factory().calculator({'x': 1})",
        "other.other({'x': 1})",
    ):
        normalized = canonicalize_generated_code(code_text=raw_code, allowed_tools=allowed_tools)
        assert normalized.parse_error is None
