"""Helpers for prompt-based structured output and JSON schema validation."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from design_research_agents.contracts.llm import (
    LLMBadResponseError,
    LLMMessage,
    LLMRequest,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolSpec

_JSON_SCHEMA_DRAFT = "2020-12"


@dataclass(slots=True, frozen=True)
class StructuredOutputResult:
    """Normalized structured output result from prompt+validate flows."""

    response: LLMResponse
    parsed: object
    attempts: int


def generate_json(
    *,
    generate_fn: Callable[[LLMRequest], LLMResponse],
    request: LLMRequest,
    schema: dict[str, object] | None,
    max_retries: int,
    extra_instructions: str | None = None,
) -> StructuredOutputResult:
    """Prompt the model for strict JSON output and validate against schema."""
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0.")

    last_error: str | None = None
    for attempt in range(max_retries + 1):
        instruction = _build_json_instruction(
            schema=schema,
            extra_instructions=extra_instructions,
            error_message=last_error,
        )
        adjusted_request = _with_instruction(request, instruction)
        response = generate_fn(adjusted_request)
        try:
            parsed = _parse_json_strict(response.text)
            _validate_json(schema, parsed)
        except Exception as exc:
            last_error = str(exc)
            continue
        return StructuredOutputResult(response=response, parsed=parsed, attempts=attempt)

    raise LLMBadResponseError(
        f"Failed to produce valid JSON after {max_retries + 1} attempt(s). Last error: {last_error}"
    )


def build_tool_call_instruction(tools: Sequence[ToolSpec]) -> str:
    """Build an instruction block describing available tools."""
    lines = [
        "You may select tools by returning JSON that matches the schema below.",
        "Available tools:",
    ]
    for tool in tools:
        schema_text = json.dumps(tool.input_schema, ensure_ascii=True, sort_keys=True)
        lines.append(f"- {tool.name}: {tool.description}")
        lines.append(f"  input_schema: {schema_text}")
    return "\n".join(lines)


def _with_instruction(request: LLMRequest, instruction: str) -> LLMRequest:
    messages = list(request.messages)
    messages.append(LLMMessage(role="system", content=instruction))
    return LLMRequest(
        messages=messages,
        model=request.model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tools=(),
        response_schema=None,
        response_format=None,
        metadata=dict(request.metadata),
        provider_options=dict(request.provider_options),
        task_profile=request.task_profile,
    )


def _build_json_instruction(
    *,
    schema: dict[str, object] | None,
    extra_instructions: str | None,
    error_message: str | None,
) -> str:
    lines = [
        "Return only valid JSON as your entire response.",
        "Do not include markdown, comments, or trailing text.",
    ]
    if schema is not None:
        schema_payload = json.dumps(schema, ensure_ascii=True, sort_keys=True, indent=2)
        lines.append(f"JSON schema (draft {_JSON_SCHEMA_DRAFT}):")
        lines.append(schema_payload)
    if extra_instructions:
        lines.append(extra_instructions)
    if error_message:
        lines.append("The previous response was invalid.")
        lines.append(f"Error: {error_message}")
        lines.append("Fix the JSON and return only the corrected payload.")
    return "\n".join(lines)


def _parse_json_strict(text: str) -> Any:
    normalized = text.strip()
    if not normalized:
        raise ValueError("Empty response; expected JSON payload.")
    return json.loads(normalized)


def _validate_json(schema: dict[str, object] | None, parsed: object) -> None:
    if schema is None:
        return
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError(
            "The 'jsonschema' package is required for structured output validation. "
            "Install with: pip install -e ."
        ) from exc
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(parsed), key=lambda err: err.path)
    if errors:
        message = errors[0].message
        path = "/".join(str(part) for part in errors[0].path)
        if path:
            message = f"{message} (at {path})"
        raise ValueError(message)
