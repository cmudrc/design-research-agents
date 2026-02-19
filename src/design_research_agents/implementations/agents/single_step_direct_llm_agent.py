"""Direct LLM agent composed from workflow building blocks."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.contracts.agent import Agent
from design_research_agents.contracts.execution import ExecutionResult
from design_research_agents.contracts.llm import LLMClient, LLMRequest, LLMResponse
from design_research_agents.contracts.workflow import LogicStep, WorkflowStepResult
from design_research_agents.implementations.shared.agent_internal.direct_llm_agent_helpers import (
    build_success_result,
    coerce_provider_options,
    extract_max_tokens,
    extract_messages,
    extract_response_schema,
    extract_temperature,
    generate_response,
    merge_provider_options,
)
from design_research_agents.implementations.shared.agent_internal.execution_context import (
    finish_agent_execution,
    prepare_agent_execution,
)
from design_research_agents.implementations.shared.agent_internal.model_resolution import (
    resolve_agent_model,
)
from design_research_agents.implementations.shared.agent_internal.workflow_first_envelope import (
    build_workflow_first_output,
)
from design_research_agents.tracing import (
    Tracer,
    finish_model_call,
    start_model_call,
)
from design_research_agents.workflow import Workflow


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
            ValueError: If max token configuration is invalid.
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
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Run one direct model call and return normalized workflow-first output.

        Args:
            prompt: Prompt text for the run.
            request_id: Optional caller-provided request id for tracing.
            dependencies: Optional dependency payload mapping.

        Returns:
            Final agent result payload.

        Raises:
            Exception: Raised when execution fails.
        """
        execution_context = prepare_agent_execution(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
            agent_name="SingleStepDirectLLMAgent",
            tracer=self._tracer,
        )
        self.workflow = self._build_workflow()

        try:
            workflow_result = self.workflow.run(
                {
                    "normalized_input": execution_context.normalized_input,
                    "request_id": execution_context.request_id,
                },
                execution_mode="sequential",
                failure_policy="skip_dependents",
                request_id=f"{execution_context.request_id}:single_step_direct",
                dependencies=execution_context.dependencies,
            )
            if not workflow_result.success:
                _raise_workflow_failure(workflow_result)
            finalize_step = workflow_result.step_results.get("finalize")
            if not isinstance(finalize_step, WorkflowStepResult) or not finalize_step.success:
                _raise_workflow_failure(workflow_result)
            assert isinstance(finalize_step, WorkflowStepResult)

            finalized = finalize_step.output
            model_response = finalized.get("model_response")
            if not isinstance(model_response, LLMResponse):
                raise TypeError("Direct single-step workflow missing LLMResponse payload.")

            base_output = finalized.get("output")
            base_metadata = finalized.get("metadata")
            result_output = (
                dict(base_output)
                if isinstance(base_output, Mapping)
                else {
                    "model": model_response.model,
                    "model_text": model_response.text,
                }
            )
            metadata = (
                dict(base_metadata)
                if isinstance(base_metadata, Mapping)
                else {
                    "request_id": execution_context.request_id,
                    "dependency_keys": sorted(execution_context.dependencies.keys()),
                }
            )
            metadata["request_id"] = execution_context.request_id
            metadata["dependency_keys"] = sorted(execution_context.dependencies.keys())
            output = build_workflow_first_output(
                base_output=result_output,
                workflow_result=workflow_result,
                final_output=model_response.text,
            )
            result = ExecutionResult(
                output=output,
                success=workflow_result.success,
                tool_results=[],
                model_response=model_response,
                metadata=metadata,
            )
        except Exception as exc:
            finish_agent_execution(trace_scope=execution_context.trace_scope, error=str(exc))
            raise

        finish_agent_execution(trace_scope=execution_context.trace_scope, result=result)
        return result

    def _build_workflow(self) -> Workflow:
        """Build one workflow graph for direct LLM execution.

        Returns:
            Workflow configured for prepare, call, and finalize stages.
        """
        return Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_mode="schema",
            steps=[
                LogicStep(step_id="prepare_request", handler=self._prepare_request_step),
                LogicStep(
                    step_id="call_model",
                    handler=self._call_model_step,
                    dependencies=("prepare_request",),
                ),
                LogicStep(
                    step_id="finalize",
                    handler=self._finalize_step,
                    dependencies=("prepare_request", "call_model"),
                ),
            ],
            default_execution_mode="sequential",
            default_failure_policy="skip_dependents",
        )

    def _prepare_request_step(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Prepare model/messages/request payload for one workflow run.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Prepared request payload consumed by downstream workflow steps.

        Raises:
            TypeError: If required schema-mode input payloads are missing or invalid.
        """
        inputs = context.get("inputs")
        if not isinstance(inputs, Mapping):
            raise TypeError("Direct single-step workflow requires schema input mapping.")
        normalized_input = inputs.get("normalized_input")
        request_id_value = inputs.get("request_id")
        if not isinstance(normalized_input, Mapping):
            raise TypeError("normalized_input must be a mapping.")
        request_id_text = str(request_id_value) if request_id_value is not None else None

        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )
        messages, message_source = extract_messages(
            input_payload=normalized_input,
            default_system_prompt=self._default_system_prompt,
        )
        llm_request = LLMRequest(
            messages=messages,
            model=resolved_model,
            temperature=extract_temperature(
                input_payload=normalized_input,
                default_value=self._temperature,
            ),
            max_tokens=extract_max_tokens(
                input_payload=normalized_input,
                default_value=self._max_tokens,
            ),
            response_schema=extract_response_schema(normalized_input),
            metadata={
                "request_id": request_id_text,
                "agent": "SingleStepDirectLLMAgent",
                "message_source": message_source,
            },
            provider_options=merge_provider_options(
                default_provider_options=self._provider_options,
                raw_provider_options=normalized_input.get("provider_options"),
            ),
        )
        return {
            "resolved_model": resolved_model,
            "messages": list(messages),
            "message_source": message_source,
            "message_count": len(messages),
            "llm_request": llm_request,
            "normalized_input": dict(normalized_input),
        }

    def _call_model_step(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Call model with prepared request payload.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Mapping containing the resolved model response payload.

        Raises:
            TypeError: If prepared dependency payloads are missing or invalid.
            Exception: Propagated when model invocation fails.
        """
        prepare_output = _dependency_output(context=context, step_id="prepare_request")
        resolved_model = prepare_output.get("resolved_model")
        raw_messages = prepare_output.get("messages")
        llm_request = prepare_output.get("llm_request")
        if not isinstance(resolved_model, str) or not isinstance(llm_request, LLMRequest):
            raise TypeError("Prepared request payload is invalid.")

        model_span_id = start_model_call(
            model=resolved_model,
            messages=list(raw_messages) if isinstance(raw_messages, list) else [],
            params=llm_request,
            metadata={
                "agent": "SingleStepDirectLLMAgent",
                "message_source": prepare_output.get("message_source", "prompt"),
            },
        )
        try:
            llm_response = generate_response(self._llm_client, llm_request)
        except Exception as exc:
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            raise

        finish_model_call(model_span_id, response=llm_response)
        return {
            "llm_response": llm_response,
        }

    def _finalize_step(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Build final direct-agent output and metadata from workflow step outputs.

        Args:
            context: Workflow step execution context payload.

        Returns:
            Finalized output/metadata mapping for workflow result projection.

        Raises:
            TypeError: If prepared dependency payloads are missing or invalid.
        """
        prepare_output = _dependency_output(context=context, step_id="prepare_request")
        call_output = _dependency_output(context=context, step_id="call_model")

        llm_response = call_output.get("llm_response")
        llm_request = prepare_output.get("llm_request")
        if not isinstance(llm_response, LLMResponse) or not isinstance(llm_request, LLMRequest):
            raise TypeError("Finalize step missing LLM request/response payload.")

        run_result = build_success_result(
            llm_response=llm_response,
            request_id=str(llm_request.metadata.get("request_id") or ""),
            dependencies={},
            message_source=str(prepare_output.get("message_source", "prompt")),
            message_count=_int_or_default(prepare_output.get("message_count"), default=0),
            llm_request=llm_request,
        )
        return {
            "output": dict(run_result.output),
            "metadata": dict(run_result.metadata),
            "model_response": llm_response,
        }


def _dependency_output(*, context: Mapping[str, object], step_id: str) -> dict[str, object]:
    """Extract dependency step output payload from workflow step context.

    Args:
        context: Workflow step execution context payload.
        step_id: Dependency step identifier to retrieve.

    Returns:
        Normalized dependency output mapping, or an empty mapping when unavailable.
    """
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return {}
    step_result = dependency_results.get(step_id)
    if not isinstance(step_result, Mapping):
        return {}
    output = step_result.get("output")
    if isinstance(output, Mapping):
        return dict(output)
    return {}


def _int_or_default(value: object, *, default: int) -> int:
    """Return integer value when coercible; otherwise return the provided default.

    Args:
        value: Candidate integer-like payload.
        default: Fallback integer value when coercion fails.

    Returns:
        Parsed integer or fallback default value.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _raise_workflow_failure(workflow_result: ExecutionResult) -> None:
    """Raise deterministic exceptions for failed workflow step outcomes.

    Args:
        workflow_result: Aggregated workflow runtime result.

    Returns:
        None.

    Raises:
        ValueError: If a failed step reported a concrete string error.
        RuntimeError: If workflow failed without a concrete step error message.
    """
    for step_id in workflow_result.execution_order:
        step_result = workflow_result.step_results.get(step_id)
        if not isinstance(step_result, WorkflowStepResult):
            continue
        if step_result.success:
            continue
        step_error = step_result.error
        if isinstance(step_error, str) and step_error.strip():
            raise ValueError(step_error)
        raise RuntimeError(f"Direct single-step workflow step '{step_id}' failed.")
    raise RuntimeError("Direct single-step workflow execution failed.")


__all__ = [
    "SingleStepDirectLLMAgent",
]
