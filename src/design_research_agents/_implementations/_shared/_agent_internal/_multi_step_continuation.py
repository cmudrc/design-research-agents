"""Shared continuation-decision helpers for multi-step agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents._contracts._termination import (
    SOURCE_GUARDRAIL,
    SOURCE_INVALID_PAYLOAD,
    SOURCE_MODEL,
)
from design_research_agents._implementations._shared._agent_internal._input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents._implementations._shared._agent_internal._multi_step_common import (
    build_continue_prompt,
    extract_continuation_thought,
    has_observation,
)
from design_research_agents._implementations._shared._agent_internal._prompt_alternatives import (
    AlternativesPromptTarget,
    inject_alternatives_into_prompt_pair,
)
from design_research_agents._implementations._shared._agent_internal._response_schemas import (
    clone_response_schema,
)
from design_research_agents._tracing import (
    emit_continuation_decision,
    emit_guardrail_decision,
    finish_model_call,
    start_model_call,
)


def llm_should_continue(
    *,
    llm_client: LLMClient,
    prompt: str,
    memory: Sequence[Mapping[str, object]],
    step_index: int,
    max_steps: int,
    model: str,
    alternatives_prompt_target: AlternativesPromptTarget,
    alternatives_text: str,
    retrieved_context: str,
    continuation_system_prompt: str,
    continuation_user_prompt_template: str,
    continuation_response_schema: dict[str, object],
    continuation_memory_tail_items: int,
    alternatives_section_label: str,
    agent_name: str,
) -> tuple[bool, str, str, LLMResponse | None]:
    """Ask the model whether execution should continue to the next step.

    Args:
        llm_client: LLM client used for continuation inference.
        prompt: User prompt text.
        memory: Current loop memory entries.
        step_index: Zero-based step index.
        max_steps: Effective max-step limit.
        model: Model identifier for the call.
        alternatives_prompt_target: Prompt target for alternatives injection.
        alternatives_text: Alternatives text block injected into prompts.
        retrieved_context: Retrieved memory context text.
        continuation_system_prompt: Continuation controller system prompt.
        continuation_user_prompt_template: Continuation controller user template.
        continuation_response_schema: Response schema used for continuation parsing.
        continuation_memory_tail_items: Memory tail size for continuation prompt rendering.
        alternatives_section_label: Label used for alternatives prompt section.
        agent_name: Agent name used for tracing/provider options.

    Returns:
        Tuple of continuation decision, reason, source, and model response.

    Raises:
        Exception: Propagates model client errors from ``LLMClient.chat``.
    """
    del max_steps
    system_prompt = continuation_system_prompt
    user_prompt = build_continue_prompt(
        prompt=prompt,
        memory=memory,
        step_number=step_index + 1,
        prompt_template=continuation_user_prompt_template,
        memory_tail_items=continuation_memory_tail_items,
        retrieved_context=retrieved_context,
    )
    system_prompt, user_prompt = inject_alternatives_into_prompt_pair(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        section_label=alternatives_section_label,
        alternatives_text=alternatives_text,
        target=alternatives_prompt_target,
    )
    messages = [
        LLMMessage(role="system", content=system_prompt),
        LLMMessage(role="user", content=user_prompt),
    ]
    llm_params = LLMChatParams(
        response_schema=clone_response_schema(continuation_response_schema),
        provider_options={
            "agent": agent_name,
            "phase": "continuation",
        },
    )
    model_span_id = start_model_call(
        model=model,
        messages=messages,
        params=llm_params,
        metadata={
            "agent": agent_name,
            "phase": "continuation",
        },
    )
    try:
        response = llm_client.chat(messages, model=model, params=llm_params)
    except Exception as exc:
        finish_model_call(model_span_id, error=str(exc), model=model)
        raise
    finish_model_call(model_span_id, response=response)

    parsed = _parse_json_mapping(response.text)
    if parsed is not None and isinstance(parsed.get("continue"), bool):
        if step_index == 0 and not bool(parsed["continue"]) and not has_observation(memory):
            emit_guardrail_decision(
                guardrail="continuation_first_step",
                decision="override_continue",
                reason="first-step guardrail",
                details={"step": step_index + 1},
            )
            emit_continuation_decision(
                step=step_index + 1,
                should_continue=True,
                reason="first-step guardrail",
                source=SOURCE_GUARDRAIL,
            )
            return True, "first-step guardrail", SOURCE_GUARDRAIL, response
        thought = extract_continuation_thought(parsed)
        emit_continuation_decision(
            step=step_index + 1,
            should_continue=bool(parsed["continue"]),
            reason=thought,
            source=SOURCE_MODEL,
        )
        return bool(parsed["continue"]), thought, SOURCE_MODEL, response

    emit_guardrail_decision(
        guardrail="continuation_output",
        decision="reject",
        reason="invalid continuation payload",
        details={"step": step_index + 1},
    )
    emit_continuation_decision(
        step=step_index + 1,
        should_continue=False,
        reason="invalid continuation payload",
        source=SOURCE_INVALID_PAYLOAD,
    )
    return False, "invalid continuation payload", SOURCE_INVALID_PAYLOAD, response


__all__ = [
    "llm_should_continue",
]
