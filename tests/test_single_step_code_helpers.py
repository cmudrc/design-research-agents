from __future__ import annotations

from design_research_agents.agent.internal.code_tool_agent_parsing import (
    AllowedTool,
    compile_default_allowed_tools,
    extract_allowed_tools,
    extract_python_code,
    match_fenced_code_block,
    normalize_allowed_tools,
)
from design_research_agents.agent.internal.input_parsing import (
    extract_boolean,
    extract_positive_int,
)
from design_research_agents.contracts.tools import ToolSpec


def _runtime_specs() -> dict[str, ToolSpec]:
    return {
        "calculator": ToolSpec(
            name="calculator",
            description="compute",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
        "text.word_count": ToolSpec(
            name="text.word_count",
            description="words",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        ),
    }


def test_extract_python_code_prefers_supported_fences() -> None:
    assert match_fenced_code_block("no fence") is None
    assert match_fenced_code_block("```js\nx=1\n```") is None
    assert match_fenced_code_block("```python\nx=1\n```") == "x=1\n"
    assert extract_python_code("```py\nx = 2\n``` trailing") == "x = 2"
    assert extract_python_code(" print('x') ") == "print('x')"


def test_numeric_and_boolean_extractors_follow_defaults() -> None:
    payload = {"max_tool_calls": 4, "bad": 0, "flag": False, "bool_like": True}
    assert extract_positive_int(input_payload=payload, key="max_tool_calls", default_value=2) == 4
    assert extract_positive_int(input_payload=payload, key="bad", default_value=2) == 2
    assert extract_positive_int(input_payload=payload, key="bool_like", default_value=2) == 2
    assert extract_boolean(input_payload=payload, key="flag", default_value=True) is False
    assert extract_boolean(input_payload=payload, key="missing", default_value=True) is True


def test_normalize_and_compile_allowed_tools_from_runtime_specs() -> None:
    specs = _runtime_specs()
    normalized = normalize_allowed_tools(
        raw_tools=[
            {
                "tool_name": "calculator",
                "description": "custom",
                "tool_input": {"expression": "1+1"},
            },
            {"name": "calculator", "description": "last-wins"},
            {"tool_name": "unknown"},
            "bad-entry",
        ],
        runtime_specs=specs,
    )
    assert len(normalized) == 1
    assert normalized[0].tool_name == "calculator"
    assert normalized[0].description == "last-wins"

    compiled_default = compile_default_allowed_tools(runtime_specs=specs, default_tools=None)
    assert {tool.tool_name for tool in compiled_default} == {
        "calculator",
        "text.word_count",
    }

    compiled_from_input = compile_default_allowed_tools(
        runtime_specs=specs,
        default_tools=[{"tool_name": "text.word_count"}],
    )
    assert [tool.tool_name for tool in compiled_from_input] == ["text.word_count"]


def test_extract_allowed_tools_clones_payloads() -> None:
    defaults = [
        AllowedTool(
            tool_name="calculator",
            description="compute",
            input_schema={"type": "object"},
            default_tool_input={"expression": "1+1"},
        )
    ]
    extracted, source = extract_allowed_tools(default_allowed_tools=defaults)
    assert source == "init_default"
    assert extracted[0].tool_name == "calculator"
    assert extracted[0] is not defaults[0]
    assert extracted[0].default_tool_input == {"expression": "1+1"}
