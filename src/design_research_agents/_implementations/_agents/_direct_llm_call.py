"""Direct one-shot LLM call composed from workflow building blocks.

This example intentionally uses the same Workflow/Step abstractions as "full" agents,
but collapses the runtime down to a single model call (no tools, no multi-step planning).
It is useful as:
- a minimal smoke-test for a backend/client,
- a pedagogical example of the workflow-first envelope,
- a building block for higher-level agents that want a thin "call model" primitive.
"""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents._contracts._delegate import Delegate
from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._llm import LLMClient, LLMRequest, LLMResponse
from design_research_agents._contracts._workflow import LogicStep, WorkflowStepResult
from design_research_agents._implementations._shared._agent_internal._direct_llm_agent_helpers import (
    build_success_result,
    coerce_provider_options,
    extract_max_tokens,
    extract_messages,
    extract_response_schema,
    extract_temperature,
    generate_response,
    merge_provider_options,
)
from design_research_agents._implementations._shared._agent_internal._execution_context import (
    resolve_agent_execution_context,
)
from design_research_agents._implementations._shared._agent_internal._model_resolution import (
    resolve_agent_model,
)
from design_research_agents._implementations._shared._agent_internal._workflow_first_envelope import (
    build_workflow_first_output,
)
from design_research_agents._tracing import (
    Tracer,
    finish_model_call,
    start_model_call,
)
from design_research_agents.workflow import CompiledExecution, Workflow


class DirectLLMCall(Delegate):
    """One-shot direct model call with no tool runtime.

    Design choices:

    - Uses a small Workflow with three LogicSteps (prepare, call, finalize) so the trace
      mirrors multi-step agents.
    - Keeps defaults (system prompt, temperature, max_tokens, provider_options) on the agent,
      but allows per-run overrides via ``normalized_input``.
    """

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
        # Validate max_tokens early so misconfiguration fails at construction time
        # rather than after a long workflow run.
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when provided.")

        # Core dependencies and default generation parameters.
        self._llm_client = llm_client
        self._default_system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

        # Tracing is optional; if unset, prepare_agent_execution will create a no-op scope.
        self._tracer = tracer

        # Normalize provider options once at init so downstream merging is predictable.
        self._provider_options = coerce_provider_options(provider_options) if provider_options is not None else {}

        # Stored for introspection/debugging; the workflow is rebuilt per-run.
        self.workflow: Workflow | None = None

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """One direct model call and return normalized workflow-first output."""
        return self.compile(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def compile(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile one direct model call into a bound workflow execution."""
        execution_context = resolve_agent_execution_context(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        )

        # Build a fresh workflow graph for this run.
        # (Safe even if the agent instance is reused across runs, but note: `self.workflow`
        # is stateful, so this class is not designed for concurrent runs on the same instance.)
        self.workflow = self._build_workflow()

        def _finalize(workflow_result: ExecutionResult) -> ExecutionResult:
            # Run the workflow with just the minimal inputs required for step handlers.
            # Workflow success means "no step failure that caused overall failure"
            # but we still require the finalize step output to build a valid agent result.
            if not workflow_result.success:
                _raise_workflow_failure(workflow_result)

            # Even if the overall workflow succeeded, finalize may have been skipped
            # (for example, if a dependency failed and was skipped). Treat this as failure
            # because we cannot produce a well-formed ExecutionResult without finalize output.
            finalize_step = workflow_result.step_results.get("finalize")
            if not isinstance(finalize_step, WorkflowStepResult) or not finalize_step.success:
                _raise_workflow_failure(workflow_result)
            assert isinstance(finalize_step, WorkflowStepResult)

            # The finalize step returns:
            # - "output": mapping projected into the workflow-first envelope
            # - "metadata": extra bookkeeping fields for callers
            # - "model_response": the raw LLMResponse for debugging and downstream reuse
            finalized = finalize_step.output
            model_response = finalized.get("model_response")
            if not isinstance(model_response, LLMResponse):
                raise TypeError("Direct call workflow missing LLMResponse payload.")

            # Optional enrichment: downstream steps may provide an explicit output/metadata mapping.
            # If they do not, fall back to a small, consistent payload derived from the model response.
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

            # Ensure request_id and dependency_keys are always present, even if finalize provided metadata.
            metadata["request_id"] = execution_context.request_id
            metadata["dependency_keys"] = sorted(execution_context.dependencies.keys())

            # Build the "workflow-first" envelope, which:
            # - preserves a structured representation of step outcomes
            # - includes a single final_output string for simple consumers
            output = build_workflow_first_output(
                base_output=result_output,
                workflow_result=workflow_result,
                final_output=model_response.text,
            )

            # Construct the normalized agent-level execution result.
            result = ExecutionResult(
                output=output,
                success=workflow_result.success,
                tool_results=[],
                model_response=model_response,
                metadata=metadata,
            )
            return result

        return CompiledExecution(
            workflow=self.workflow,
            input={
                "normalized_input": execution_context.normalized_input,
                "request_id": execution_context.request_id,
            },
            request_id=execution_context.request_id,
            workflow_request_id=f"{execution_context.request_id}:direct_call",
            dependencies=execution_context.dependencies,
            delegate_name="DirectLLMCall",
            finalize=_finalize,
            tracer=self._tracer,
            trace_input=execution_context.normalized_input,
        )

    def _build_workflow(self) -> Workflow:
        """Build one workflow graph for direct LLM execution.

        Returns:
            Workflow configured for prepare, call, and finalize stages.
        """
        # Note: tool_runtime=None makes this a pure-LLM workflow (no tool execution).
        # input_schema is intentionally permissive here; per-step handlers validate what they need.
        return Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_schema={"type": "object"},
            steps=[
                # Step 1: normalize inputs, resolve model, and build an LLMRequest.
                LogicStep(step_id="prepare_request", handler=self._prepare_request_step),
                # Step 2: call the model client and capture an LLMResponse.
                LogicStep(
                    step_id="call_model",
                    handler=self._call_model_step,
                    dependencies=("prepare_request",),
                ),
                # Step 3: project response into a stable output/metadata shape.
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
        # The workflow runtime passes a standardized `context` mapping containing:
        # - inputs: the workflow run input payload
        # - dependency_results: outputs/errors from upstream steps
        inputs = context.get("inputs")
        if not isinstance(inputs, Mapping):
            raise TypeError("Direct call workflow requires schema input mapping.")

        # Pull out the normalized input and request id created by prepare_agent_execution.
        normalized_input = inputs.get("normalized_input")
        request_id_value = inputs.get("request_id")
        if not isinstance(normalized_input, Mapping):
            raise TypeError("normalized_input must be a mapping.")
        request_id_text = str(request_id_value) if request_id_value is not None else None

        # Resolve which model to use for this agent run (may depend on client defaults/config).
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )

        # Extract messages from normalized_input:
        # - if user provided explicit messages, use them
        # - otherwise synthesize messages from prompt + optional system prompt
        messages, message_source = extract_messages(
            input_payload=normalized_input,
            default_system_prompt=self._default_system_prompt,
        )

        # Build a single LLMRequest object, merging defaults with per-run overrides.
        # Provider options are merged last so callers can override backend-specific knobs.
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
                "agent": "DirectLLMCall",
                "message_source": message_source,
            },
            provider_options=merge_provider_options(
                default_provider_options=self._provider_options,
                raw_provider_options=normalized_input.get("provider_options"),
            ),
        )

        # Return a step output mapping; downstream steps access this via dependency_results.
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
        # Dependency outputs are stored under context["dependency_results"] keyed by step_id.
        prepare_output = _dependency_output(context=context, step_id="prepare_request")

        # Validate the specific fields we rely on; this keeps failures clear and localized.
        resolved_model = prepare_output.get("resolved_model")
        raw_messages = prepare_output.get("messages")
        llm_request = prepare_output.get("llm_request")
        if not isinstance(resolved_model, str) or not isinstance(llm_request, LLMRequest):
            raise TypeError("Prepared request payload is invalid.")

        # Start a tracing span for the model call (inputs + params + metadata).
        # This produces a model_span_id used to close the span deterministically.
        model_span_id = start_model_call(
            model=resolved_model,
            messages=list(raw_messages) if isinstance(raw_messages, list) else [],
            params=llm_request,
            metadata={
                "agent": "DirectLLMCall",
                "message_source": prepare_output.get("message_source", "prompt"),
                "step_id": "call_model",
            },
        )
        try:
            # Single backend call: this is where network / provider errors propagate.
            llm_response = generate_response(self._llm_client, llm_request)
        except Exception as exc:
            # Ensure the span is closed with an error so traces are not left dangling.
            finish_model_call(model_span_id, error=str(exc), model=resolved_model)
            raise

        # Close the trace span with the provider response payload.
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
        # Pull outputs from the two dependencies we require.
        prepare_output = _dependency_output(context=context, step_id="prepare_request")
        call_output = _dependency_output(context=context, step_id="call_model")

        # Ensure we have both request and response objects; without them we cannot normalize output.
        llm_response = call_output.get("llm_response")
        llm_request = prepare_output.get("llm_request")
        if not isinstance(llm_response, LLMResponse) or not isinstance(llm_request, LLMRequest):
            raise TypeError("Finalize step missing LLM request/response payload.")

        # Convert the raw response into a stable "success" result shape.
        # This centralizes output normalization (model text, metadata fields, message stats, etc.).
        run_result = build_success_result(
            llm_response=llm_response,
            request_id=str(llm_request.metadata.get("request_id") or ""),
            dependencies={},
            message_source=str(prepare_output.get("message_source", "prompt")),
            message_count=_int_or_default(prepare_output.get("message_count"), default=0),
            llm_request=llm_request,
        )

        # Return both:
        # - structured output/metadata (dicts)
        # - the raw model response object for the outer `run()` method to attach to ExecutionResult
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
    # Missing or malformed dependency payloads are treated as empty mappings so
    # step handlers can raise targeted validation errors for required fields.
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

    Notes:
        - bool is a subclass of int in Python; we handle it explicitly for clarity.
        - strings are accepted only if `int(value)` succeeds.
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
    # Preserve the first concrete failed step in execution order. This keeps error
    # reporting deterministic even when multiple downstream steps are skipped/failed.
    for step_id in workflow_result.execution_order:
        step_result = workflow_result.step_results.get(step_id)
        if not isinstance(step_result, WorkflowStepResult):
            continue
        if step_result.success:
            continue
        step_error = step_result.error
        if isinstance(step_error, str) and step_error.strip():
            raise ValueError(step_error)
        raise RuntimeError(f"Direct call workflow step '{step_id}' failed.")
    raise RuntimeError("Direct call workflow execution failed.")


__all__ = [
    "DirectLLMCall",
]
