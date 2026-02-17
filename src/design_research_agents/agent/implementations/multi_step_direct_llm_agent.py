"""Multi-step direct-LLM controller with iterative CONTINUE/STOP decisions.

This agent does not invoke external tools. Instead, each controller step emits
an internal action:
- ``CONTINUE`` with refined partial progress, or
- ``STOP`` with the final response text.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

from design_research_agents.agent.internal.input_parsing import (
    extract_positive_int as _extract_positive_int,
)
from design_research_agents.agent.internal.input_parsing import (
    extract_prompt as _extract_prompt,
)
from design_research_agents.agent.internal.input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.multi_step_common import build_step_prompt
from design_research_agents.agent.internal.prompt_overrides import resolve_prompt_text
from design_research_agents.agent.internal.response_schemas import (
    build_multi_step_direct_controller_response_schema,
    clone_response_schema,
)
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.tracing import (
    Tracer,
    emit_continuation_decision,
    emit_guardrail_decision,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)


@dataclass(slots=True, frozen=True)
class _ControllerDecision:
    """One parsed controller action for a direct-response step."""

    decision: str
    content: str
    final_output: str | None
    reason: str
    source: str


class MultiStepDirectLLMAgent(Agent):
    """Agent that iterates internal direct-response controller decisions."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        max_steps: int = 5,
        controller_system_prompt: str | None = None,
        controller_user_prompt_template: str | None = None,
        step_memory_tail_items: int = 8,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a multi-step direct-response controller agent.

        Args:
            llm_client: LLM client used for each controller step.
            max_steps: Maximum number of controller steps.
            controller_system_prompt: Optional controller system prompt override.
            controller_user_prompt_template: Optional controller user prompt template override.
            step_memory_tail_items: Memory tail size rendered into each controller step prompt.
            tracer: Optional explicit tracer dependency.
        """
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1.")
        if step_memory_tail_items < 1:
            raise ValueError("step_memory_tail_items must be >= 1.")

        self._llm_client = llm_client
        self._tracer = tracer
        self._max_steps = max_steps
        self._controller_system_prompt = resolve_prompt_text(
            override=controller_system_prompt,
            default_prompt_name="multi_step_direct_controller_system",
            field_name="controller_system_prompt",
        )
        self._controller_user_prompt_template = resolve_prompt_text(
            override=controller_user_prompt_template,
            default_prompt_name="multi_step_direct_controller_user",
            field_name="controller_user_prompt_template",
        )
        self._step_memory_tail_items = step_memory_tail_items
        self._controller_response_schema = build_multi_step_direct_controller_response_schema()

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Run iterative CONTINUE/STOP controller steps until termination."""
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        trace_scope = start_trace_run(
            agent_name="MultiStepDirectLLMAgent",
            request_id=resolved_request_id,
            input_payload=normalized_input,
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )
        prompt = _extract_prompt(normalized_input)
        max_steps = _extract_positive_int(
            input_payload=normalized_input,
            key="max_steps",
            default_value=self._max_steps,
        )
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )

        memory: list[dict[str, object]] = [{"kind": "task", "prompt": prompt}]
        step_outputs: list[dict[str, object]] = []
        final_output = ""
        terminated_reason = "max_steps_reached"
        last_model_response: LLMResponse | None = None

        for step_index in range(max_steps):
            step_number = step_index + 1
            user_prompt = build_step_prompt(
                prompt=prompt,
                memory=memory,
                step_number=step_number,
                prompt_template=self._controller_user_prompt_template,
                memory_tail_items=self._step_memory_tail_items,
            )
            messages = [
                LLMMessage(role="system", content=self._controller_system_prompt),
                LLMMessage(role="user", content=user_prompt),
            ]
            llm_params = LLMChatParams(
                response_schema=clone_response_schema(self._controller_response_schema),
                provider_options={
                    "agent": "MultiStepDirectLLMAgent",
                    "phase": "controller_step",
                },
            )
            model_span_id = start_model_call(
                model=resolved_model,
                messages=messages,
                params=llm_params,
                metadata={
                    "agent": "MultiStepDirectLLMAgent",
                    "phase": "controller_step",
                },
            )
            try:
                llm_response = self._llm_client.chat(
                    messages,
                    model=resolved_model,
                    params=llm_params,
                )
            except Exception as exc:
                finish_model_call(model_span_id, error=str(exc), model=resolved_model)
                finish_trace_run(trace_scope, error=str(exc))
                raise
            finish_model_call(model_span_id, response=llm_response)
            last_model_response = llm_response

            parsed_decision = _parse_controller_decision(llm_response.text)
            if parsed_decision is None:
                emit_guardrail_decision(
                    guardrail="direct_controller_output",
                    decision="fallback",
                    reason="invalid controller JSON",
                    details={"step": step_number},
                )
                parsed_decision = _fallback_controller_decision(
                    raw_text=llm_response.text,
                    step_index=step_index,
                    max_steps=max_steps,
                )

            if parsed_decision.decision == "CONTINUE":
                final_output = parsed_decision.content
                emit_continuation_decision(
                    step=step_number,
                    should_continue=True,
                    reason=parsed_decision.reason,
                    source=parsed_decision.source,
                )
                step_outputs.append(
                    {
                        "step": step_number,
                        "decision": "CONTINUE",
                        "content": parsed_decision.content,
                        "reason": parsed_decision.reason,
                        "source": parsed_decision.source,
                    }
                )
                memory.append(
                    {
                        "kind": "continue",
                        "step": step_number,
                        "content": parsed_decision.content,
                        "reason": parsed_decision.reason,
                        "source": parsed_decision.source,
                    }
                )
                continue

            final_output = parsed_decision.final_output or parsed_decision.content
            terminated_reason = f"stop:{parsed_decision.source}"
            emit_continuation_decision(
                step=step_number,
                should_continue=False,
                reason=parsed_decision.reason,
                source=parsed_decision.source,
            )
            step_outputs.append(
                {
                    "step": step_number,
                    "decision": "STOP",
                    "final_output": final_output,
                    "reason": parsed_decision.reason,
                    "source": parsed_decision.source,
                }
            )
            memory.append(
                {
                    "kind": "stop",
                    "step": step_number,
                    "final_output": final_output,
                    "reason": parsed_decision.reason,
                    "source": parsed_decision.source,
                }
            )
            break

        result = AgentResult(
            output={
                "final_output": final_output,
                "steps_executed": len(step_outputs),
                "step_outputs": step_outputs,
                "memory": memory,
                "terminated_reason": terminated_reason,
            },
            success=True,
            tool_results=[],
            model_response=last_model_response,
            metadata={
                "request_id": resolved_request_id,
                "dependency_keys": sorted(resolved_dependencies.keys()),
                "controller_steps": list(step_outputs),
                "config": {
                    "max_steps": max_steps,
                    "step_memory_tail_items": self._step_memory_tail_items,
                },
            },
        )
        finish_trace_run(trace_scope, result=result)
        return result

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Emit a deterministic stream wrapper around ``run``."""
        result = self.run(prompt, request_id=request_id, dependencies=dependencies)
        delta_text = result.model_response.text if result.model_response is not None else ""
        yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=result)


def _parse_controller_decision(raw_text: str) -> _ControllerDecision | None:
    parsed = _parse_json_mapping(raw_text)
    if parsed is not None:
        raw_decision = parsed.get("decision")
        if isinstance(raw_decision, str):
            normalized_decision = raw_decision.strip().upper()
            if normalized_decision in {"CONTINUE", "STOP"}:
                content = str(parsed.get("content", "")).strip()
                final_output = parsed.get("final_output")
                return _ControllerDecision(
                    decision=normalized_decision,
                    content=content,
                    final_output=(str(final_output).strip() if final_output is not None else None),
                    reason=str(parsed.get("reason", content or "model decision")),
                    source="model",
                )

        raw_continue = parsed.get("continue")
        if isinstance(raw_continue, bool):
            content = str(parsed.get("thought", parsed.get("content", ""))).strip()
            return _ControllerDecision(
                decision="CONTINUE" if raw_continue else "STOP",
                content=content,
                final_output=(
                    str(parsed.get("final_output")).strip()
                    if parsed.get("final_output") is not None
                    else None
                ),
                reason=content or "model decision",
                source="model_legacy",
            )

    normalized_text = raw_text.strip()
    upper_text = normalized_text.upper()
    if upper_text.startswith("CONTINUE"):
        return _ControllerDecision(
            decision="CONTINUE",
            content=_strip_action_prefix(normalized_text, "CONTINUE"),
            final_output=None,
            reason="text action prefix",
            source="text_fallback",
        )
    if upper_text.startswith("STOP"):
        stripped = _strip_action_prefix(normalized_text, "STOP")
        return _ControllerDecision(
            decision="STOP",
            content=stripped,
            final_output=stripped,
            reason="text action prefix",
            source="text_fallback",
        )
    return None


def _fallback_controller_decision(
    *,
    raw_text: str,
    step_index: int,
    max_steps: int,
) -> _ControllerDecision:
    if step_index + 1 >= max_steps:
        return _ControllerDecision(
            decision="STOP",
            content=raw_text,
            final_output=raw_text,
            reason="fallback max-steps stop",
            source="fallback",
        )
    return _ControllerDecision(
        decision="CONTINUE",
        content=raw_text,
        final_output=None,
        reason="fallback continue",
        source="fallback",
    )


def _strip_action_prefix(text: str, action: str) -> str:
    raw_value = text[len(action) :].strip()
    if raw_value.startswith(":"):
        raw_value = raw_value[1:].strip()
    return raw_value
