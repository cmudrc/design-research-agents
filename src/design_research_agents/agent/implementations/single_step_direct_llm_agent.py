"""Direct LLM agent that forwards requests without any tool orchestration.

The agent builds a minimal chat payload, calls the configured ``LLMClient``,
and returns the response as a standard ``ExecutionResult``.
"""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.agent.internal.direct_llm_agent_helpers import (
    build_success_result,
    coerce_provider_options,
    extract_max_tokens,
    extract_messages,
    extract_response_schema,
    extract_temperature,
    generate_response,
    merge_provider_options,
)
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.llm import (
    LLMClient,
    LLMMessage,
    LLMRequest,
)
from design_research_agents.tracing import (
    Tracer,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


class SingleStepDirectLLMAgent(Agent):
    """Agent that performs one direct model call with no tool runtime."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        provider_options: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a direct-LLM agent with optional default generation args.

        Args:
            llm_client: LLM client used for prompt execution.
            system_prompt: Optional default system prompt.
            temperature: Optional default sampling temperature.
            max_tokens: Optional default output-token cap.
            provider_options: Optional default backend-specific options.
            tracer: Optional explicit tracer dependency.

        Raises:
            Exception: Raised when execution fails.
        """
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when provided.")

        self._llm_client = llm_client
        self._default_system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._tracer = tracer
        self._provider_options = (
            coerce_provider_options(provider_options) if provider_options is not None else {}
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Run one direct model call and return normalized ``ExecutionResult`` output.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.

        Raises:
            Exception: Raised when execution fails.
        """
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        resolved_model, messages, message_source, llm_request = self._prepare_request(
            normalized_input,
            request_id=resolved_request_id,
        )
        trace_scope = start_trace_run(
            agent_name="SingleStepDirectLLMAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )
        model_span_id = start_model_call(
            model=resolved_model,
            messages=messages,
            params=llm_request,
            metadata={
                "agent": "SingleStepDirectLLMAgent",
                "message_source": message_source,
            },
        )
        try:
            llm_response = generate_response(self._llm_client, llm_request)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            finish_trace_run(trace_scope, error=str(exc))
            raise

        finish_model_call(model_span_id, response=llm_response)
        result = build_success_result(
            llm_response=llm_response,
            request_id=resolved_request_id,
            dependencies=resolved_dependencies,
            message_source=message_source,
            message_count=len(messages),
            llm_request=llm_request,
        )
        finish_trace_run(trace_scope, result=result)
        return result

    def _prepare_request(
        self,
        input_payload: Mapping[str, object],
        *,
        request_id: str | None,
    ) -> tuple[str, list[LLMMessage], str, LLMRequest]:
        """Resolve model/messages/params into one reusable request payload.

        Args:
            input_payload: Normalized run input payload mapping.
            request_id: Optional request identifier used in metadata.

        Returns:
            Tuple of resolved model, messages, message source, and chat params.
        """
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )
        messages, message_source = extract_messages(
            input_payload=input_payload,
            default_system_prompt=self._default_system_prompt,
        )
        llm_request = LLMRequest(
            messages=messages,
            model=resolved_model,
            temperature=extract_temperature(
                input_payload=input_payload,
                default_value=self._temperature,
            ),
            max_tokens=extract_max_tokens(
                input_payload=input_payload,
                default_value=self._max_tokens,
            ),
            response_schema=extract_response_schema(input_payload),
            metadata={
                "request_id": request_id,
                "agent": "SingleStepDirectLLMAgent",
                "message_source": message_source,
            },
            provider_options=merge_provider_options(
                default_provider_options=self._provider_options,
                raw_provider_options=input_payload.get("provider_options"),
            ),
        )
        return resolved_model, messages, message_source, llm_request


__all__ = [
    "SingleStepDirectLLMAgent",
]
