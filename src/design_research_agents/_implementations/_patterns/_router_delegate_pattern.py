"""Reusable intent/agent-routing orchestration chunk."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from design_research_agents._contracts._delegate import Delegate, ExecutionResult
from design_research_agents._contracts._llm import LLMClient
from design_research_agents._contracts._termination import (
    TERMINATED_ROUTING_FAILURE,
    TERMINATED_UNKNOWN_ALTERNATIVE,
)
from design_research_agents._contracts._tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents._contracts._workflow import (
    DelegateStep,
    DelegateTarget,
    LogicStep,
    WorkflowExecutionMode,
    WorkflowFailurePolicy,
)
from design_research_agents._runtime._common._delegate_invocation import invoke_delegate
from design_research_agents._runtime._patterns import (
    MODE_ROUTER_DELEGATE,
    WorkflowBudgetTracker,
    attach_runtime_metadata,
    build_compiled_pattern_execution,
    build_pattern_execution_result,
    normalize_request_id_prefix,
    resolve_pattern_run_context,
)
from design_research_agents._skills import (
    SkillsConfig,
    SkillsContext,
    SkillsToolRuntimeAdapter,
    merge_skills_metadata,
    resolve_skills_context,
)
from design_research_agents._tracing import Tracer
from design_research_agents.workflow import CompiledExecution
from design_research_agents.workflow.workflow import Workflow

from .._shared._agent_internal._agent_routing_runtime_adapter import (
    AgentRoutingToolRuntimeAdapter,
)
from .._shared._agent_internal._input_parsing import (
    extract_prompt as _extract_prompt,
)
from .._shared._agent_internal._json_action_step_runner import (
    JsonActionStepRunner,
)
from .._shared._agent_internal._prompt_overrides import (
    validate_prompt_text,
)
from .._shared._agent_internal._run_options import (
    normalize_input_payload,
)

_ROUTING_FAILURE_ROUTE = "__routing_failure__"
_UNKNOWN_ALTERNATIVE_ROUTE = "__unknown_alternative__"


def _build_alternative_step_ids(alternative_names: Sequence[str]) -> dict[str, str]:
    """Return stable workflow step ids for router alternatives."""
    step_ids: dict[str, str] = {}
    used_step_ids: set[str] = set()
    for index, alternative_name in enumerate(alternative_names, start=1):
        sanitized_name = "".join(
            character.lower() if character.isalnum() else "_" for character in alternative_name.strip()
        ).strip("_")
        if not sanitized_name:
            sanitized_name = f"alternative_{index}"
        candidate_step_id = f"agent_routing_delegate_{sanitized_name}"
        suffix = 2
        while candidate_step_id in used_step_ids:
            candidate_step_id = f"agent_routing_delegate_{sanitized_name}_{suffix}"
            suffix += 1
        used_step_ids.add(candidate_step_id)
        step_ids[alternative_name] = candidate_step_id
    return step_ids


def _resolve_delegate_workflow_for_diagram(
    *,
    delegate: DelegateTarget,
    prompt: str,
    request_id: str,
    dependencies: Mapping[str, object],
) -> Workflow | None:
    """Return a compiled workflow for delegate diagram expansion when available."""
    direct_workflow = delegate if isinstance(delegate, Workflow) else None
    if direct_workflow is not None:
        return direct_workflow

    existing_workflow = getattr(delegate, "workflow", None)
    if isinstance(existing_workflow, Workflow):
        return existing_workflow

    compile_callable = getattr(delegate, "compile", None)
    if not callable(compile_callable):
        return None

    try:
        compiled = compile_callable(
            prompt,
            request_id=f"{request_id}:diagram",
            dependencies=dependencies,
        )
    except Exception:
        fallback_workflow = getattr(delegate, "workflow", None)
        return fallback_workflow if isinstance(fallback_workflow, Workflow) else None

    compiled_workflow = getattr(compiled, "workflow", None)
    if isinstance(compiled_workflow, Workflow):
        return compiled_workflow
    fallback_workflow = getattr(delegate, "workflow", None)
    return fallback_workflow if isinstance(fallback_workflow, Workflow) else None


@dataclass(slots=True, kw_only=True)
class _RoutingExecutionState:
    """Mutable state shared between routing workflow callbacks."""

    router_result: ExecutionResult | None = None
    """Result produced by the router selection step, if it ran."""

    delegated_result: ExecutionResult | None = None
    """Result produced by the selected delegate step, if it ran."""

    router_tool_results: list[ToolResult] = field(default_factory=list)
    """Tool results produced by router selection, including optional skill activation."""


class _AlternativeDelegateRunner:
    """Workflow-runner wrapper that preserves router pattern delegate semantics."""

    def __init__(
        self,
        *,
        selected_name: str,
        delegate: DelegateTarget,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        state: _RoutingExecutionState,
        budget_tracker: WorkflowBudgetTracker,
        runtime_tool_specs: Mapping[str, ToolSpec],
    ) -> None:
        self._selected_name = selected_name
        self._delegate = delegate
        self._state = state
        self._budget_tracker = budget_tracker
        self._runtime_tool_specs = runtime_tool_specs
        self.workflow = _resolve_delegate_workflow_for_diagram(
            delegate=delegate,
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        )

    def run(
        self,
        *,
        context: Mapping[str, object] | None = None,
        execution_mode: WorkflowExecutionMode = "dag",
        failure_policy: WorkflowFailurePolicy = "skip_dependents",
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Invoke the wrapped delegate while keeping the outer workflow step green."""
        normalized_context = dict(context or {})
        prompt_value = normalized_context.get("prompt")
        if not isinstance(prompt_value, str) or not prompt_value.strip():
            raise ValueError("Router delegate runner requires a non-empty prompt in context.")
        resolved_request_id = str(request_id) if request_id is not None else f"router_delegate:{self._selected_name}"
        delegate_invocation = invoke_delegate(
            delegate=self._delegate,
            prompt=prompt_value,
            step_context=normalized_context,
            request_id=resolved_request_id,
            execution_mode=execution_mode,
            failure_policy=failure_policy,
            dependencies=dict(dependencies or {}),
        )
        self._state.delegated_result = delegate_invocation.result
        delegated_result = self._state.delegated_result
        self._budget_tracker.add_model_response(delegated_result.model_response)
        self._budget_tracker.add_tool_results(
            tool_results=delegated_result.tool_results,
            tool_specs=self._runtime_tool_specs,
        )
        return ExecutionResult(
            success=True,
            output={
                "status": "delegated",
                "selected_name": self._selected_name,
                "delegated_success": delegated_result.success,
                "delegate_type": delegate_invocation.delegate_type,
            },
            metadata={
                "selected_alternative": self._selected_name,
                "delegate_type": delegate_invocation.delegate_type,
            },
        )


class _RoutingWorkflowCallbacks:
    """Workflow callback bundle for router-selection and fallback steps."""

    def __init__(
        self,
        *,
        router_agent: JsonActionStepRunner,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        budget_tracker: WorkflowBudgetTracker,
        alternative_step_ids: Mapping[str, str],
        routing_failure_step_id: str,
        unknown_alternative_step_id: str,
        state: _RoutingExecutionState,
        router_agent_after_activation: JsonActionStepRunner | None = None,
    ) -> None:
        """Store dependencies used by workflow callback methods.

        Args:
            router_agent: Router agent used in selection step.
            prompt: Routed user prompt.
            request_id: Resolved request id.
            dependencies: Normalized dependency mapping.
            budget_tracker: Budget tracker collecting model/tool usage.
            alternative_step_ids: Workflow step ids keyed by alternative name.
            routing_failure_step_id: Fallback step id used when routing selection fails.
            unknown_alternative_step_id: Fallback step id used for unknown alternatives.
            state: Mutable callback state sink.
            router_agent_after_activation: Optional second-pass router used after one
                successful ``skills.activate`` call.
        """
        self._router_agent = router_agent
        self._prompt = prompt
        self._request_id = request_id
        self._dependencies = dependencies
        self._budget_tracker = budget_tracker
        self._alternative_step_ids = dict(alternative_step_ids)
        self._routing_failure_step_id = routing_failure_step_id
        self._unknown_alternative_step_id = unknown_alternative_step_id
        self._state = state
        self._router_agent_after_activation = router_agent_after_activation

    def run_selection(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Execute router-model selection step.

        Args:
            context: Workflow step context payload (unused).

        Returns:
            Step output describing routing status and selected delegate name.
        """
        del context
        router_result = self._router_agent.run(
            self._prompt,
            request_id=f"{self._request_id}:agent_routing_router",
            dependencies=self._dependencies,
        )
        router_tool_results = list(router_result.tool_results)
        self._budget_tracker.add_model_response(router_result.model_response)
        if (
            self._router_agent_after_activation is not None
            and _selected_tool_name_from_result(router_result) == "skills.activate"
        ):
            activated_prompt = _append_activated_skill_context(
                prompt=self._prompt,
                tool_results=router_tool_results,
            )
            router_result = self._router_agent_after_activation.run(
                activated_prompt,
                request_id=f"{self._request_id}:agent_routing_router_after_skill",
                dependencies=self._dependencies,
            )
            router_tool_results.extend(router_result.tool_results)
            self._budget_tracker.add_model_response(router_result.model_response)
        self._state.router_result = router_result
        self._state.router_tool_results = router_tool_results
        if not router_result.success:
            return {
                "status": TERMINATED_ROUTING_FAILURE,
                "route": _ROUTING_FAILURE_ROUTE,
                "selected_step_id": self._routing_failure_step_id,
                "routing": router_result.metadata.get("routing", {}),
            }

        selected_name = _extract_selected_name_from_router_output(router_result.output)
        selected_step_id = self._alternative_step_ids.get(selected_name)
        if selected_step_id is None:
            return {
                "status": "selected",
                "selected_name": selected_name,
                "route": _UNKNOWN_ALTERNATIVE_ROUTE,
                "selected_step_id": self._unknown_alternative_step_id,
                "routing": router_result.metadata.get("routing", {}),
            }
        return {
            "status": "selected",
            "selected_name": selected_name,
            "route": selected_name,
            "selected_step_id": selected_step_id,
            "routing": router_result.metadata.get("routing", {}),
        }

    def run_routing_failure(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Emit a stable failure payload for router-model selection failures."""
        selection_output = _extract_selection_output(context) or {}
        return {
            "status": TERMINATED_ROUTING_FAILURE,
            "routing": selection_output.get("routing", {}),
        }

    def run_unknown_alternative(self, context: Mapping[str, object]) -> Mapping[str, object]:
        """Emit a stable failure payload when the router selects an undeclared alternative."""
        selection_output = _extract_selection_output(context)
        if selection_output is None:
            return {
                "status": TERMINATED_ROUTING_FAILURE,
                "routing": {},
            }
        return {
            "status": TERMINATED_UNKNOWN_ALTERNATIVE,
            "selected_name": str(selection_output.get("selected_name", "")).strip(),
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
    available_alternatives: Sequence[str],
    workflow_payload: Mapping[str, object],
    workflow_artifacts: Sequence[object],
    skills_context: SkillsContext | None,
    tool_results: Sequence[ToolResult],
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
        available_alternatives: Declared delegate names available to the router.
        workflow_payload: Serialized workflow payload for this routing run.
        workflow_artifacts: Normalized workflow artifact entries.
        skills_context: Optional resolved Agent Skills context.
        tool_results: Tool results already produced during routing selection.

    Returns:
        Execution result carrying normalized routing failure metadata.
    """
    failure = build_pattern_execution_result(
        success=False,
        final_output={},
        terminated_reason=terminated_reason,
        details={
            "selected_alternative": None,
            "available_alternatives": list(available_alternatives),
            "delegated_result": {},
            "router": router_result.metadata.get("routing", {}),
        },
        workflow_payload=dict(workflow_payload),
        artifacts=list(workflow_artifacts),
        request_id=request_id,
        dependencies=dependencies,
        mode=MODE_ROUTER_DELEGATE,
        metadata=merge_skills_metadata(
            metadata={"stage": stage, "routing": router_result.metadata.get("routing", {})},
            skills_context=skills_context,
            tool_results=tool_results,
        ),
        tool_results=list(tool_results),
        model_response=router_result.model_response,
        error=error,
    )
    return attach_runtime_metadata(
        agent_result=failure,
        requested_mode=MODE_ROUTER_DELEGATE,
        resolved_mode=MODE_ROUTER_DELEGATE,
        budget_metadata=budget_tracker.as_metadata(),
        extra_metadata=None,
    )


class RouterDelegatePattern(Delegate):
    """Routing/delegation pattern built on workflow primitives."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        alternatives: Mapping[str, DelegateTarget],
        alternative_descriptions: Mapping[str, str] | None = None,
        router_system_prompt: str | None = None,
        router_user_prompt_template: str | None = None,
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        skills: SkillsConfig | None = None,
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
            skills: Optional Agent Skills configuration.
            tracer: Optional tracer used for run-level instrumentation.

        Raises:
            ValueError: If no valid route alternatives are supplied.
        """
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._tracer = tracer
        self._skills_context = resolve_skills_context(skills)
        self.workflow: Workflow | None = None
        self._agent_routing_runtime: dict[str, object] | None = None
        self._default_request_id_prefix = normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._alternatives = {
            name.strip(): agent for name, agent in alternatives.items() if isinstance(name, str) and name.strip()
        }
        if not self._alternatives:
            raise ValueError("alternatives must include at least one non-empty route key.")

        self._alternative_descriptions = {
            name.strip(): description.strip()
            for name, description in (alternative_descriptions or {}).items()
            if isinstance(name, str) and name.strip() and isinstance(description, str) and description.strip()
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
        """Execute one intent-routing orchestration run."""
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
        """Compile one intent-routing orchestration run."""
        run_context = resolve_pattern_run_context(
            default_request_id_prefix=self._default_request_id_prefix,
            default_dependencies=self._default_dependencies,
            request_id=request_id,
            dependencies=dependencies,
        )
        normalized_input = normalize_input_payload(prompt)
        resolved_prompt = _extract_prompt(normalized_input)
        workflow = self._build_workflow(
            resolved_prompt,
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
        )
        runtime = self._agent_routing_runtime or {}
        budget_tracker = runtime.get("budget_tracker")
        execution_state = runtime.get("execution_state")
        if not isinstance(budget_tracker, WorkflowBudgetTracker) or not isinstance(
            execution_state, _RoutingExecutionState
        ):
            raise RuntimeError("Agent routing runtime state is unavailable before workflow execution.")
        return build_compiled_pattern_execution(
            workflow=workflow,
            pattern_name="RouterDelegatePattern",
            request_id=run_context.request_id,
            dependencies=run_context.dependencies,
            tracer=self._tracer,
            input_payload={"prompt": resolved_prompt, "mode": MODE_ROUTER_DELEGATE},
            workflow_request_id=f"{run_context.request_id}:router_delegate_workflow",
            finalize=lambda workflow_result: self._finalize_agent_routing_result(
                workflow_result=workflow_result,
                budget_tracker=budget_tracker,
                execution_state=execution_state,
                request_id=run_context.request_id,
                dependencies=run_context.dependencies,
            ),
        )

    def _build_workflow(
        self,
        prompt: str,
        *,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> Workflow:
        """Build the routing workflow for one resolved run context."""
        budget_tracker = WorkflowBudgetTracker()
        alternative_step_ids = _build_alternative_step_ids(tuple(self._alternatives))
        routing_failure_step_id = "agent_routing_failure"
        unknown_alternative_step_id = "agent_routing_unknown_alternative"
        routing_tool_runtime = AgentRoutingToolRuntimeAdapter(
            alternatives=self._alternatives,
            descriptions=self._alternative_descriptions,
        )
        automatic_activation_enabled = bool(
            self._skills_context is not None
            and self._skills_context.config.allow_automatic_activation
            and self._skills_context.discovered_skill_names
        )
        router_skills_context = self._skills_context
        router_agent_after_activation: JsonActionStepRunner | None = None
        routed_tool_runtime: ToolRuntime = routing_tool_runtime
        if automatic_activation_enabled and self._skills_context is not None:
            routed_tool_runtime = SkillsToolRuntimeAdapter(
                wrapped_runtime=routing_tool_runtime,
                skills_context=self._skills_context,
            )
            router_agent_after_activation = JsonActionStepRunner(
                llm_client=self._llm_client,
                tool_runtime=routing_tool_runtime,
                system_prompt=self._router_system_prompt,
                user_prompt_template=self._router_user_prompt_template,
                allowed_tools=tuple(sorted(self._alternatives)),
                skills_context=_pinned_only_skills_context(self._skills_context),
                tracer=self._tracer,
            )
        router_agent = JsonActionStepRunner(
            llm_client=self._llm_client,
            tool_runtime=routed_tool_runtime,
            system_prompt=self._router_system_prompt,
            user_prompt_template=self._router_user_prompt_template,
            allowed_tools=tuple(sorted(self._alternatives)),
            skills_context=router_skills_context,
            tracer=self._tracer,
        )
        runtime_tool_specs: dict[str, ToolSpec] = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        execution_state = _RoutingExecutionState()
        callbacks = _RoutingWorkflowCallbacks(
            router_agent=router_agent,
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
            budget_tracker=budget_tracker,
            alternative_step_ids=alternative_step_ids,
            routing_failure_step_id=routing_failure_step_id,
            unknown_alternative_step_id=unknown_alternative_step_id,
            state=execution_state,
            router_agent_after_activation=router_agent_after_activation,
        )
        routed_delegate_steps = [
            DelegateStep(
                step_id=step_id,
                delegate=_AlternativeDelegateRunner(
                    selected_name=alternative_name,
                    delegate=self._alternatives[alternative_name],
                    prompt=prompt,
                    request_id=request_id,
                    dependencies=dependencies,
                    state=execution_state,
                    budget_tracker=budget_tracker,
                    runtime_tool_specs=runtime_tool_specs,
                ),
                dependencies=("agent_routing_selection",),
                prompt=prompt,
            )
            for alternative_name, step_id in alternative_step_ids.items()
        ]
        workflow = Workflow(
            tool_runtime=None,
            tracer=self._tracer,
            input_schema={"type": "object"},
            base_context={"prompt": prompt},
            steps=[
                LogicStep(
                    step_id="agent_routing_selection",
                    handler=callbacks.run_selection,
                    route_map={
                        **{name: (step_id,) for name, step_id in alternative_step_ids.items()},
                        _ROUTING_FAILURE_ROUTE: (routing_failure_step_id,),
                        _UNKNOWN_ALTERNATIVE_ROUTE: (unknown_alternative_step_id,),
                    },
                ),
                *routed_delegate_steps,
                LogicStep(
                    step_id=routing_failure_step_id,
                    dependencies=("agent_routing_selection",),
                    handler=callbacks.run_routing_failure,
                ),
                LogicStep(
                    step_id=unknown_alternative_step_id,
                    dependencies=("agent_routing_selection",),
                    handler=callbacks.run_unknown_alternative,
                ),
            ],
        )
        self.workflow = workflow
        self._agent_routing_runtime = {
            "budget_tracker": budget_tracker,
            "execution_state": execution_state,
        }
        return workflow

    def _run_agent_routing(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Router-selection workflow and delegated agent execution.

        Args:
            prompt: User prompt to route.
            request_id: Resolved request id for this orchestration run.
            dependencies: Normalized dependency mapping for delegates.

        Returns:
            Pattern result containing routing decision and delegate output.

        Raises:
            RuntimeError: If internal workflow invariants are violated.
        """
        workflow = self._build_workflow(
            prompt,
            request_id=request_id,
            dependencies=dependencies,
        )
        runtime = self._agent_routing_runtime or {}
        budget_tracker = runtime.get("budget_tracker")
        execution_state = runtime.get("execution_state")
        if not isinstance(budget_tracker, WorkflowBudgetTracker) or not isinstance(
            execution_state, _RoutingExecutionState
        ):
            raise RuntimeError("Agent routing runtime state is unavailable before workflow execution.")

        workflow_result = workflow.run(
            input={},
            execution_mode="sequential",
            failure_policy="skip_dependents",
            request_id=f"{request_id}:router_delegate_workflow",
            dependencies=dependencies,
        )
        return self._finalize_agent_routing_result(
            workflow_result=workflow_result,
            budget_tracker=budget_tracker,
            execution_state=execution_state,
            request_id=request_id,
            dependencies=dependencies,
        )

    def _finalize_agent_routing_result(
        self,
        *,
        workflow_result: ExecutionResult,
        budget_tracker: WorkflowBudgetTracker,
        execution_state: _RoutingExecutionState,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> ExecutionResult:
        """Build the final routed result from a workflow execution."""
        workflow_payload = workflow_result.to_dict()
        raw_workflow_artifacts = workflow_result.output.get("artifacts")
        workflow_artifacts: list[object] = (
            list(raw_workflow_artifacts)
            if isinstance(raw_workflow_artifacts, Sequence) and not isinstance(raw_workflow_artifacts, (str, bytes))
            else []
        )

        if not workflow_result.success:
            raise RuntimeError("Agent routing workflow graph execution failed.")

        router_result = execution_state.router_result
        if router_result is None:
            raise RuntimeError("Agent routing selection step did not produce a router result.")

        selection_step = workflow_result.step_results.get("agent_routing_selection")
        selection_output = selection_step.output if selection_step is not None else {}
        selected_step_id = str(selection_output.get("selected_step_id", "")).strip()
        delegate_step = workflow_result.step_results.get(selected_step_id) if selected_step_id else None
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
                available_alternatives=sorted(self._alternatives.keys()),
                workflow_payload=workflow_payload,
                workflow_artifacts=workflow_artifacts,
                skills_context=self._skills_context,
                tool_results=execution_state.router_tool_results,
            )

        selected_name = str(selection_output.get("selected_name", "")).strip()
        if delegate_output.get("status") == TERMINATED_UNKNOWN_ALTERNATIVE or selected_name not in self._alternatives:
            return _build_routing_failure_result(
                error=f"Agent routing selected unknown agent alternative '{selected_name}'.",
                request_id=request_id,
                dependencies=dependencies,
                router_result=router_result,
                budget_tracker=budget_tracker,
                stage="agent_routing_selection",
                terminated_reason=TERMINATED_UNKNOWN_ALTERNATIVE,
                available_alternatives=sorted(self._alternatives.keys()),
                workflow_payload=workflow_payload,
                workflow_artifacts=workflow_artifacts,
                skills_context=self._skills_context,
                tool_results=execution_state.router_tool_results,
            )

        delegated_result = execution_state.delegated_result
        if delegated_result is None:
            return _build_routing_failure_result(
                error="Agent routing delegate execution did not run.",
                request_id=request_id,
                dependencies=dependencies,
                router_result=router_result,
                budget_tracker=budget_tracker,
                stage=selected_step_id or "agent_routing_delegate",
                terminated_reason=TERMINATED_ROUTING_FAILURE,
                available_alternatives=sorted(self._alternatives.keys()),
                workflow_payload=workflow_payload,
                workflow_artifacts=workflow_artifacts,
                skills_context=self._skills_context,
                tool_results=execution_state.router_tool_results,
            )

        router_delegate_metadata = {
            "routing": router_result.metadata.get("routing", {}),
            "selected_alternative": selected_name,
            "available_alternatives": sorted(self._alternatives.keys()),
        }
        combined_tool_results = [
            *execution_state.router_tool_results,
            *list(delegated_result.tool_results),
        ]

        delegated_output = dict(delegated_result.output)
        delegated_final_output = delegated_output.get("final_output")
        if not isinstance(delegated_final_output, Mapping):
            delegated_final_output = dict(delegated_output)

        result = build_pattern_execution_result(
            success=delegated_result.success,
            final_output=dict(delegated_final_output),
            terminated_reason=delegated_result.terminated_reason or "completed",
            details={
                "selected_alternative": selected_name,
                "available_alternatives": sorted(self._alternatives.keys()),
                "delegated_result": dict(delegated_output),
                "router": router_result.metadata.get("routing", {}),
            },
            workflow_payload=workflow_payload,
            artifacts=list(workflow_artifacts),
            request_id=request_id,
            dependencies=dependencies,
            mode=MODE_ROUTER_DELEGATE,
            metadata=merge_skills_metadata(
                metadata={
                    **dict(delegated_result.metadata),
                    "router_delegate": router_delegate_metadata,
                },
                skills_context=self._skills_context,
                tool_results=combined_tool_results,
            ),
            tool_results=combined_tool_results,
            model_response=delegated_result.model_response,
            error=delegated_result.error,
        )
        return attach_runtime_metadata(
            agent_result=result,
            requested_mode=MODE_ROUTER_DELEGATE,
            resolved_mode=MODE_ROUTER_DELEGATE,
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
    return ""


def _selected_tool_name_from_result(result: ExecutionResult) -> str:
    """Extract the selected tool name from one router-agent result."""
    return _extract_selected_name_from_router_output(result.output)


def _append_activated_skill_context(
    *,
    prompt: str,
    tool_results: Sequence[ToolResult],
) -> str:
    """Append one activated skill block to the router prompt when available."""
    for tool_result in tool_results:
        if tool_result.tool_name != "skills.activate" or not tool_result.ok:
            continue
        payload = tool_result.result_dict()
        name = str(payload.get("name", "")).strip()
        description = str(payload.get("description", "")).strip()
        instructions = str(payload.get("instructions", "")).strip()
        skill_root = str(payload.get("skill_root", "")).strip()
        compatibility_raw = payload.get("compatibility")
        compatibility = (
            ", ".join(str(item) for item in compatibility_raw)
            if isinstance(
                compatibility_raw,
                list,
            )
            else "(none)"
        )
        skill_block = "\n".join(
            [
                "Activated routing skill:",
                f'<active_skill name="{name}" root="{skill_root}">',
                f"description: {description}",
                f"compatibility: {compatibility}",
                "instructions:",
                instructions,
                "</active_skill>",
                "Use the activated routing skill when choosing the best alternative.",
            ]
        ).strip()
        if skill_block:
            return f"{prompt.rstrip()}\n\n{skill_block}"
    return prompt


def _pinned_only_skills_context(skills_context: SkillsContext) -> SkillsContext:
    """Return one derived skills context that suppresses automatic activation."""
    return SkillsContext(
        config=SkillsConfig(
            project_root=skills_context.config.project_root,
            extra_paths=skills_context.config.extra_paths,
            pinned_skills=skills_context.config.pinned_skills,
            catalog_prompt_target=skills_context.config.catalog_prompt_target,
            allow_automatic_activation=False,
        ),
        catalog=skills_context.catalog,
        pinned_skills=skills_context.pinned_skills,
    )


__all__ = [
    "RouterDelegatePattern",
]
