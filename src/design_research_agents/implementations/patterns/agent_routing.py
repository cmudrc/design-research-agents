"""Reusable intent/agent-routing orchestration chunk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.termination import (
    TERMINATED_ROUTING_FAILURE,
    TERMINATED_UNKNOWN_ALTERNATIVE,
)
from design_research_agents.contracts.tools import ToolRuntime, ToolSpec
from design_research_agents.contracts.workflow import LogicStep, WorkflowDelegate
from design_research_agents.implementations.agents.multi_step_agent import MultiStepAgent
from design_research_agents.tracing import Tracer
from design_research_agents.workflow.workflow import Workflow

from ..shared.agent_internal.agent_routing_runtime_adapter import (
    AgentRoutingToolRuntimeAdapter,
)
from ..shared.agent_internal.input_parsing import (
    extract_prompt as _extract_prompt,
)
from ..shared.agent_internal.prompt_overrides import (
    validate_prompt_text,
)
from ..shared.agent_internal.run_options import (
    normalize_input_payload,
)
from ..shared.workflow_internal import (
    WorkflowBudgetTracker,
    attach_runtime_metadata,
    build_pattern_failure_result,
    execute_pattern_with_trace,
    normalize_request_id_prefix,
    resolve_pattern_run_context,
)
from ..shared.workflow_internal.delegate_invocation import invoke_delegate


@dataclass(slots=True)
class _RoutingExecutionState:
    """Mutable state shared between routing workflow callbacks."""

    router_result: ExecutionResult | None = None
    """Result produced by the router selection step, if it ran."""

    delegated_result: ExecutionResult | None = None
    """Result produced by the selected delegate step, if it ran."""


class _RoutingWorkflowCallbacks:
    """Workflow callback bundle for selection and delegate execution steps."""

    def __init__(
        self,
        *,
        pattern: RouterPattern,
        router_agent: MultiStepAgent,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        budget_tracker: WorkflowBudgetTracker,
        runtime_tool_specs: Mapping[str, ToolSpec],
        state: _RoutingExecutionState,
    ) -> None:
        """Store dependencies used by workflow callback methods.

        Args:
            pattern: Router pattern instance owning delegate alternatives.
            router_agent: Router agent used in selection step.
            prompt: Routed user prompt.
            request_id: Resolved request id.
            dependencies: Normalized dependency mapping.
            budget_tracker: Budget tracker collecting model/tool usage.
            runtime_tool_specs: Runtime tool specs keyed by tool name.
            state: Mutable callback state sink.
        """
        self._pattern = pattern
        self._router_agent = router_agent
        self._prompt = prompt
        self._request_id = request_id
        self._dependencies = dependencies
        self._budget_tracker = budget_tracker
        self._runtime_tool_specs = runtime_tool_specs
        self._state = state

    def run_selection(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Execute router-model selection step.

        Args:
            context: Workflow step context payload (unused).

        Returns:
            Step output describing routing status and selected delegate name.
        """
        del context
        self._state.router_result = self._router_agent.run(
            self._prompt,
            request_id=f"{self._request_id}:agent_routing_router",
            dependencies=self._dependencies,
        )
        router_result = self._state.router_result
        self._budget_tracker.add_model_response(router_result.model_response)
        if not router_result.success:
            return {
                "status": TERMINATED_ROUTING_FAILURE,
                "routing": router_result.metadata.get("routing", {}),
            }

        selected_name = _extract_selected_name_from_router_output(router_result.output)
        return {
            "status": "selected",
            "selected_name": selected_name,
            "routing": router_result.metadata.get("routing", {}),
        }

    def run_delegate(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Execute selected delegate agent based on selection-step output.

        Args:
            context: Workflow step context containing dependency step outputs.

        Returns:
            Step output describing delegation status and selected agent metadata.
        """
        selection_output = _extract_selection_output(context)
        if selection_output is None:
            return {
                "status": TERMINATED_ROUTING_FAILURE,
                "routing": {},
            }

        status = selection_output.get("status")
        if status != "selected":
            return {
                "status": TERMINATED_ROUTING_FAILURE,
                "routing": selection_output.get("routing", {}),
            }

        selected_name = str(selection_output.get("selected_name", "")).strip()
        selected_delegate = self._pattern._alternatives.get(selected_name)
        if selected_delegate is None:
            return {
                "status": TERMINATED_UNKNOWN_ALTERNATIVE,
                "selected_name": selected_name,
                "routing": selection_output.get("routing", {}),
            }

        delegate_invocation = invoke_delegate(
            delegate=selected_delegate,
            prompt=self._prompt,
            step_context=context,
            request_id=f"{self._request_id}:agent_routing:{selected_name}",
            execution_mode="sequential",
            failure_policy="skip_dependents",
            dependencies=self._dependencies,
        )
        self._state.delegated_result = delegate_invocation.result
        delegated_result = self._state.delegated_result
        self._budget_tracker.add_model_response(delegated_result.model_response)
        self._budget_tracker.add_tool_results(
            tool_results=delegated_result.tool_results,
            tool_specs=self._runtime_tool_specs,
        )
        return {
            "status": "delegated",
            "selected_name": selected_name,
            "delegated_success": delegated_result.success,
            "routing": selection_output.get("routing", {}),
        }


def _extract_selection_output(context: Mapping[str, object]) -> Mapping[str, object] | None:
    """Extract routing selection output from step context.

    Args:
        context: Delegate-step execution context mapping.

    Returns:
        Selection output mapping from ``agent_routing_selection`` when present.
    """
    dependency_results = context.get("dependency_results")
    if not isinstance(dependency_results, Mapping):
        return None
    selection_step = dependency_results.get("agent_routing_selection")
    if not isinstance(selection_step, Mapping):
        return None
    selection_output = selection_step.get("output")
    if not isinstance(selection_output, Mapping):
        return None
    return selection_output


def _build_routing_failure_result(
    *,
    error: str,
    request_id: str,
    dependencies: Mapping[str, object],
    router_result: ExecutionResult,
    budget_tracker: WorkflowBudgetTracker,
    stage: str,
    terminated_reason: str,
    workflow_payload: Mapping[str, object],
    workflow_artifacts: Sequence[object],
) -> ExecutionResult:
    """Build one attached routing failure result with stable metadata.

    Args:
        error: Human-readable failure description.
        request_id: Request id for correlation metadata.
        dependencies: Normalized dependency mapping.
        router_result: Router selection result used for carry-forward metadata.
        budget_tracker: Runtime budget tracker for aggregate metadata.
        stage: Internal routing stage where failure occurred.
        terminated_reason: Canonical termination reason.
        workflow_payload: Serialized workflow payload for this routing run.
        workflow_artifacts: Normalized workflow artifact entries.

    Returns:
        Execution result carrying normalized routing failure metadata.
    """
    failure = build_pattern_failure_result(
        error=error,
        model_response=router_result.model_response,
        request_id=request_id,
        dependencies=dependencies,
        metadata={
            "stage": stage,
            "mode": "agent_routing",
            "routing": router_result.metadata.get("routing", {}),
        },
        output={
            "terminated_reason": terminated_reason,
            "routing": router_result.metadata.get("routing", {}),
            "delegated_agent": None,
            "delegated_output": {},
            "final_output": {},
            "workflow": dict(workflow_payload),
            "artifacts": list(workflow_artifacts),
        },
    )
    return attach_runtime_metadata(
        agent_result=failure,
        requested_mode="agent_routing",
        resolved_mode="agent_routing",
        budget_metadata=budget_tracker.as_metadata(),
        extra_metadata=None,
    )


class RouterPattern(Agent):
    """Routing/delegation pattern built on workflow primitives."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        alternatives: Mapping[str, WorkflowDelegate],
        alternative_descriptions: Mapping[str, str] | None = None,
        router_system_prompt: str | None = None,
        router_user_prompt_template: str | None = None,
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and initialize workflow-native routing settings.

        Args:
            llm_client: LLM client used by the router agent.
            tool_runtime: Tool runtime used to cost/metadata-account delegated calls.
            alternatives: Mapping of route keys to delegate objects.
            alternative_descriptions: Optional descriptions used to guide routing.
            router_system_prompt: Optional override for router system prompt.
            router_user_prompt_template: Optional override for router user prompt.
            default_request_id_prefix: Optional prefix used to derive request ids.
            default_dependencies: Dependency defaults merged into each run.
            tracer: Optional tracer used for run-level instrumentation.

        Raises:
            ValueError: If no valid route alternatives are supplied.
        """
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._tracer = tracer
        self.workflow: Workflow | None = None
        self._default_request_id_prefix = normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._alternatives = {
            name.strip(): agent
            for name, agent in alternatives.items()
            if isinstance(name, str) and name.strip()
        }
        if not self._alternatives:
            raise ValueError("alternatives must include at least one non-empty route key.")

        self._alternative_descriptions = {
            name.strip(): description.strip()
            for name, description in (alternative_descriptions or {}).items()
            if isinstance(name, str)
            and name.strip()
            and isinstance(description, str)
            and description.strip()
        }
        self._router_system_prompt = (
            validate_prompt_text(
                value=router_system_prompt,
                field_name="router_system_prompt",
            )
            if router_system_prompt is not None
            else None
        )
        self._router_user_prompt_template = (
            validate_prompt_text(
                value=router_user_prompt_template,
                field_name="router_user_prompt_template",
            )
            if router_user_prompt_template is not None
            else None
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one intent-routing orchestration run.

        Args:
            prompt: User prompt to route to the best delegate.
            request_id: Optional request id for tracing and correlation.
            dependencies: Optional dependency overrides for this run.

        Returns:
            Final routed agent result with routing metadata.

        Raises:
            Exception: Propagates runtime failures from routing/delegate execution.
        """
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=self._default_request_id_prefix,
            default_dependencies=self._default_dependencies,
            request_id=request_id,
            dependencies=dependencies,
        )
        normalized_input = normalize_input_payload(prompt)
        resolved_prompt = _extract_prompt(normalized_input)
        return execute_pattern_with_trace(
            agent_name="RouterPattern",
            request_id=run_context.request_id,
            input_payload={"prompt": resolved_prompt, "mode": "agent_routing"},
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            runner=lambda: self._run_agent_routing(
                prompt=resolved_prompt,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
            ),
        )

    def _run_agent_routing(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Run router-selection workflow and delegated agent execution.

        Args:
            prompt: User prompt to route.
            request_id: Resolved request id for this orchestration run.
            dependencies: Normalized dependency mapping for delegates.

        Returns:
            Pattern result containing routing decision and delegate output.

        Raises:
            RuntimeError: If internal workflow invariants are violated.
        """
        budget_tracker = WorkflowBudgetTracker()
        routing_tool_runtime = AgentRoutingToolRuntimeAdapter(
            alternatives=self._alternatives,
            descriptions=self._alternative_descriptions,
        )
        router_agent = MultiStepAgent(
            mode="json",
            llm_client=self._llm_client,
            tool_runtime=routing_tool_runtime,
            max_steps=1,
            stop_on_step_failure=True,
            tool_calling_system_prompt=self._router_system_prompt,
            tool_calling_user_prompt_template=self._router_user_prompt_template,
            allowed_tools=tuple(sorted(self._alternatives)),
            tracer=self._tracer,
        )
        runtime_tool_specs: dict[str, ToolSpec] = {
            spec.name: spec for spec in self._tool_runtime.list_tools()
        }
        execution_state = _RoutingExecutionState()
        callbacks = _RoutingWorkflowCallbacks(
            pattern=self,
            router_agent=router_agent,
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
            budget_tracker=budget_tracker,
            runtime_tool_specs=runtime_tool_specs,
            state=execution_state,
        )

        self.workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_mode="schema",
            base_context={"prompt": prompt},
            steps=[
                LogicStep(step_id="agent_routing_selection", handler=callbacks.run_selection),
                LogicStep(
                    step_id="agent_routing_delegate",
                    dependencies=("agent_routing_selection",),
                    handler=callbacks.run_delegate,
                ),
            ],
        )
        workflow_result = self.workflow.run(
            {},
            execution_mode="sequential",
            failure_policy="skip_dependents",
            request_id=f"{request_id}:agent_routing_workflow",
            dependencies=dependencies,
        )
        workflow_payload = workflow_result.asdict()
        raw_workflow_artifacts = workflow_result.output.get("artifacts")
        workflow_artifacts: list[object] = (
            list(raw_workflow_artifacts)
            if isinstance(raw_workflow_artifacts, Sequence)
            and not isinstance(raw_workflow_artifacts, (str, bytes))
            else []
        )

        if not workflow_result.success:
            raise RuntimeError("Agent routing workflow graph execution failed.")

        router_result = execution_state.router_result
        if router_result is None:
            raise RuntimeError("Agent routing selection step did not produce a router result.")

        selection_step = workflow_result.step_results.get("agent_routing_selection")
        selection_output = selection_step.output if selection_step is not None else {}
        delegate_step = workflow_result.step_results.get("agent_routing_delegate")
        delegate_output = delegate_step.output if delegate_step is not None else {}

        if not router_result.success:
            return _build_routing_failure_result(
                error="Agent routing selection failed.",
                request_id=request_id,
                dependencies=dependencies,
                router_result=router_result,
                budget_tracker=budget_tracker,
                stage="agent_routing_selection",
                terminated_reason=TERMINATED_ROUTING_FAILURE,
                workflow_payload=workflow_payload,
                workflow_artifacts=workflow_artifacts,
            )

        selected_name = str(selection_output.get("selected_name", "")).strip()
        if (
            delegate_output.get("status") == TERMINATED_UNKNOWN_ALTERNATIVE
            or selected_name not in self._alternatives
        ):
            return _build_routing_failure_result(
                error=f"Agent routing selected unknown agent alternative '{selected_name}'.",
                request_id=request_id,
                dependencies=dependencies,
                router_result=router_result,
                budget_tracker=budget_tracker,
                stage="agent_routing_selection",
                terminated_reason=TERMINATED_UNKNOWN_ALTERNATIVE,
                workflow_payload=workflow_payload,
                workflow_artifacts=workflow_artifacts,
            )

        delegated_result = execution_state.delegated_result
        if delegated_result is None:
            return _build_routing_failure_result(
                error="Agent routing delegate execution did not run.",
                request_id=request_id,
                dependencies=dependencies,
                router_result=router_result,
                budget_tracker=budget_tracker,
                stage="agent_routing_delegate",
                terminated_reason=TERMINATED_ROUTING_FAILURE,
                workflow_payload=workflow_payload,
                workflow_artifacts=workflow_artifacts,
            )

        agent_routing_metadata = {
            "routing": router_result.metadata.get("routing", {}),
            "selected_alternative": selected_name,
            "available_alternatives": sorted(self._alternatives.keys()),
        }

        delegated_output = dict(delegated_result.output)
        delegated_output["agent_routing_selected_alternative"] = selected_name
        final_output = delegated_output.get("final_output")
        if not isinstance(final_output, Mapping):
            final_output = dict(delegated_output)

        result = ExecutionResult(
            output={
                **delegated_output,
                "delegated_agent": selected_name,
                "delegated_output": dict(delegated_output),
                "final_output": dict(final_output),
                "workflow": workflow_payload,
                "artifacts": list(workflow_artifacts),
            },
            success=delegated_result.success,
            tool_results=list(delegated_result.tool_results),
            model_response=delegated_result.model_response,
            metadata={
                **dict(delegated_result.metadata),
                "agent_routing": agent_routing_metadata,
            },
        )
        return attach_runtime_metadata(
            agent_result=result,
            requested_mode="agent_routing",
            resolved_mode="agent_routing",
            budget_metadata=budget_tracker.as_metadata(),
            extra_metadata={
                "workflow": {
                    "execution_order": list(workflow_result.execution_order),
                }
            },
        )


def _extract_selected_name_from_router_output(output: Mapping[str, object]) -> str:
    """Extract selected delegate name from multi-step router output payload.

    Args:
        output: Router output mapping containing ``step_outputs``.

    Returns:
        Normalized selected delegate key, or empty string when unavailable.
    """
    direct_tool_name = output.get("tool_name")
    if isinstance(direct_tool_name, str) and direct_tool_name.strip():
        return direct_tool_name.strip()

    raw_step_outputs = output.get("step_outputs")
    if not isinstance(raw_step_outputs, Sequence) or isinstance(raw_step_outputs, (str, bytes)):
        return ""
    for raw_step_output in reversed(raw_step_outputs):
        if not isinstance(raw_step_output, Mapping):
            continue
        tool_name = raw_step_output.get("tool_name")
        if isinstance(tool_name, str) and tool_name.strip():
            return tool_name.strip()
        raw_tool_names = raw_step_output.get("tool_names")
        if isinstance(raw_tool_names, Sequence) and not isinstance(raw_tool_names, (str, bytes)):
            for raw_name in raw_tool_names:
                if not isinstance(raw_name, str):
                    continue
                normalized_name = raw_name.strip()
                if normalized_name:
                    return normalized_name
    return ""


__all__ = [
    "RouterPattern",
]
