"""Base backend interface and shared capability enforcement helpers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass

from design_research_agents._contracts._llm import (
    BackendCapabilities,
    BackendStatus,
    EmbeddingResult,
    LLMCapabilityError,
    LLMDelta,
    LLMInvalidRequestError,
    LLMRequest,
    LLMResponse,
    ToolCall,
)
from design_research_agents._contracts._tools import ToolSpec
from design_research_agents.llm._structured_output import (
    build_tool_call_instruction,
    generate_json,
)


@dataclass(slots=True, frozen=True)
class ToolCallSchemaConfig:
    """Schema configuration for best-effort tool call extraction."""

    property_name: str = "tool_calls"
    """Field value for ``property_name``."""


class BaseLLMBackend(ABC):
    """Base backend with capability enforcement and prompt+validate helpers."""

    name: str
    kind: str
    default_model: str | None
    base_url: str | None
    config_hash: str
    max_retries: int
    model_patterns: tuple[str, ...]

    def __init__(
        self,
        *,
        name: str,
        kind: str,
        default_model: str | None = None,
        base_url: str | None = None,
        config_hash: str,
        max_retries: int = 2,
        model_patterns: Sequence[str] | None = None,
    ) -> None:
        """Initialize immutable backend identity and routing metadata.

        Args:
            name: Input value for this parameter.
            kind: Input value for this parameter.
            default_model: Input value for this parameter.
            base_url: Input value for this parameter.
            config_hash: Input value for this parameter.
            max_retries: Input value for this parameter.
            model_patterns: Input value for this parameter.
        """
        self.name = name
        self.kind = kind
        self.default_model = default_model
        self.base_url = base_url
        self.config_hash = config_hash
        self.max_retries = max_retries
        self.model_patterns = tuple(model_patterns or ())

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response while enforcing backend capability constraints.

        Args:
            request: Input value for this parameter.

        Returns:
            Computed return value.

        Raises:
            Exception: Raised when this operation cannot complete.
        """
        resolved_request = self._resolve_model(request)
        if resolved_request.tools and resolved_request.response_schema:
            raise LLMInvalidRequestError("Requests cannot specify both tools and response_schema.")

        capabilities = self.capabilities()
        if resolved_request.tools:
            if capabilities.tool_calling == "none":
                raise LLMCapabilityError(f"Backend '{self.name}' does not support tool calling.")
            if capabilities.tool_calling == "best_effort":
                return self._generate_best_effort_tool_calls(resolved_request)

        if resolved_request.response_schema or resolved_request.response_format:
            if capabilities.json_mode == "none":
                raise LLMCapabilityError(f"Backend '{self.name}' does not support JSON output.")
            if capabilities.json_mode == "prompt+validate":
                return self._generate_prompt_validated_json(resolved_request)

        return self._generate(resolved_request)

    def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Stream response deltas for the given request.

        Args:
            request: Input value for this parameter.

        Returns:
            Computed return value.

        Raises:
            Exception: Raised when this operation cannot complete.
        """
        resolved_request = self._resolve_model(request)
        if not self.capabilities().streaming:
            raise LLMCapabilityError(f"Backend '{self.name}' does not support streaming.")
        return self._stream(resolved_request)

    def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        """Return embeddings for the supplied texts (optional).

        Args:
            texts: Input value for this parameter.

        Returns:
            Computed return value.

        Raises:
            Exception: Raised when this operation cannot complete.
        """
        raise LLMCapabilityError(f"Backend '{self.name}' does not support embeddings.")

    def supports_model(self, model: str) -> bool:
        """Return whether the backend claims to support the given model id.

        Args:
            model: Input value for this parameter.

        Returns:
            Computed return value.
        """
        if not self.model_patterns:
            return True
        return any(_matches_model_pattern(model, pattern) for pattern in self.model_patterns)

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Return declared backend capabilities.

        Returns:
            Computed return value.
        """

    @abstractmethod
    def healthcheck(self) -> BackendStatus:
        """Return backend healthcheck status.

        Returns:
            Computed return value.
        """

    @abstractmethod
    def _generate(self, request: LLMRequest) -> LLMResponse:
        """Provider-specific non-streaming generation implementation.

        Args:
            request: Input value for this parameter.

        Returns:
            Computed return value.
        """

    @abstractmethod
    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        """Provider-specific streaming generation implementation.

        Args:
            request: Input value for this parameter.

        Returns:
            Computed return value.
        """

    def _resolve_model(self, request: LLMRequest) -> LLMRequest:
        """Run resolve model.

        Args:
            request: Input value for this parameter.

        Returns:
            Computed return value.

        Raises:
            Exception: Raised when this operation cannot complete.
        """
        model = request.model.strip() if request.model else self.default_model or ""
        if not model or "*" in model:
            raise LLMInvalidRequestError(f"Backend '{self.name}' requires an explicit model id.")
        if not self.supports_model(model):
            raise LLMInvalidRequestError(f"Backend '{self.name}' does not support model '{model}'.")
        if request.model == model:
            return request
        return _replace_request_model(request, model=model)

    def _generate_prompt_validated_json(self, request: LLMRequest) -> LLMResponse:
        """Run generate prompt validated json.

        Args:
            request: Input value for this parameter.

        Returns:
            Computed return value.
        """
        result = generate_json(
            generate_fn=self._generate,
            request=request,
            schema=request.response_schema,
            max_retries=self.max_retries,
            extra_instructions=None,
        )
        normalized_text = _normalize_json_text(result.parsed)
        return _merge_response(
            result.response,
            text=normalized_text,
            raw=_merge_raw(
                result.response.raw,
                {
                    "structured_output": {
                        "attempts": result.attempts + 1,
                        "parsed": result.parsed,
                    }
                },
            ),
        )

    def _generate_best_effort_tool_calls(self, request: LLMRequest) -> LLMResponse:
        """Run generate best effort tool calls.

        Args:
            request: Input value for this parameter.

        Returns:
            Computed return value.
        """
        tools = request.tools
        tool_schema = _build_tool_call_schema(tools)
        instruction = build_tool_call_instruction(tools)
        result = generate_json(
            generate_fn=self._generate,
            request=request,
            schema=tool_schema,
            max_retries=self.max_retries,
            extra_instructions=instruction,
        )
        tool_calls = _parse_tool_calls(result.parsed, tools)
        return _merge_response(
            result.response,
            tool_calls=tuple(tool_calls),
            raw=_merge_raw(
                result.response.raw,
                {
                    "structured_output": {
                        "attempts": result.attempts + 1,
                        "parsed": result.parsed,
                        "tool_calls": [asdict(call) for call in tool_calls],
                    }
                },
            ),
        )


def _replace_request_model(request: LLMRequest, *, model: str) -> LLMRequest:
    """Run replace request model.

    Args:
        request: Input value for this parameter.
        model: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return LLMRequest(
        messages=request.messages,
        model=model,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        tools=request.tools,
        response_schema=request.response_schema,
        response_format=request.response_format,
        metadata=dict(request.metadata),
        provider_options=dict(request.provider_options),
        task_profile=request.task_profile,
    )


def _merge_response(
    response: LLMResponse,
    *,
    text: str | None = None,
    tool_calls: tuple[ToolCall, ...] | None = None,
    raw: dict[str, object] | None = None,
) -> LLMResponse:
    """Run merge response.

    Args:
        response: Input value for this parameter.
        text: Input value for this parameter.
        tool_calls: Input value for this parameter.
        raw: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return LLMResponse(
        text=response.text if text is None else text,
        tool_calls=response.tool_calls if tool_calls is None else tool_calls,
        usage=response.usage,
        raw=raw if raw is not None else response.raw,
        provenance=response.provenance,
        model=response.model,
        provider=response.provider,
        finish_reason=response.finish_reason,
        latency_ms=response.latency_ms,
    )


def _merge_raw(
    current: dict[str, object] | None,
    update: dict[str, object],
) -> dict[str, object]:
    """Run merge raw.

    Args:
        current: Input value for this parameter.
        update: Input value for this parameter.

    Returns:
        Computed return value.
    """
    merged = dict(current or {})
    merged.update(update)
    return merged


def _normalize_json_text(parsed: object) -> str:
    """Run normalize json text.

    Args:
        parsed: Input value for this parameter.

    Returns:
        Computed return value.
    """
    return json.dumps(parsed, ensure_ascii=True, sort_keys=True)


def _build_tool_call_schema(tools: Sequence[ToolSpec]) -> dict[str, object]:
    """Run build tool call schema.

    Args:
        tools: Input value for this parameter.

    Returns:
        Computed return value.
    """
    tool_names = [tool.name for tool in tools]
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["tool_calls"],
        "properties": {
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["name", "arguments"],
                    "properties": {
                        "name": {"type": "string", "enum": tool_names},
                        "arguments": {"type": "object"},
                    },
                },
            }
        },
    }


def _parse_tool_calls(parsed: object, tools: Sequence[ToolSpec]) -> list[ToolCall]:
    """Run parse tool calls.

    Args:
        parsed: Input value for this parameter.
        tools: Input value for this parameter.

    Returns:
        Computed return value.

    Raises:
        Exception: Raised when this operation cannot complete.
    """
    tool_names = {tool.name for tool in tools}
    payload = parsed
    if isinstance(payload, dict) and "tool_calls" in payload:
        payload = payload.get("tool_calls")
    if isinstance(payload, dict) and "tool_name" in payload:
        payload = [
            {
                "name": payload.get("tool_name"),
                "arguments": payload.get("tool_input"),
            }
        ]
    if not isinstance(payload, list):
        raise ValueError("Parsed tool calls payload is not a list.")
    tool_calls: list[ToolCall] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError("Tool call entry must be an object.")
        name = item.get("name")
        arguments = item.get("arguments", {})
        if not isinstance(name, str) or name not in tool_names:
            raise ValueError(f"Unknown tool name '{name}'.")
        if not isinstance(arguments, dict):
            raise ValueError("Tool call arguments must be an object.")
        call_id = item.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index + 1}"
        tool_calls.append(
            ToolCall(
                name=name,
                arguments_json=_normalize_json_text(arguments),
                call_id=call_id,
            )
        )
    return tool_calls


def _matches_model_pattern(model: str, pattern: str) -> bool:
    """Run matches model pattern.

    Args:
        model: Input value for this parameter.
        pattern: Input value for this parameter.

    Returns:
        Computed return value.
    """
    if pattern == model:
        return True
    if "*" not in pattern:
        return False
    parts = pattern.split("*")
    if len(parts) == 2:
        prefix, suffix = parts
        return model.startswith(prefix) and model.endswith(suffix)
    return False
