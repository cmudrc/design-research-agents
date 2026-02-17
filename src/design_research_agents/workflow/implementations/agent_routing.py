"""Reusable intent/agent-routing orchestration chunk."""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from design_research_agents.agent.implementations.single_step_router_agent import (
    SingleStepRouterAgent,
)
from design_research_agents.agent.internal.agent_routing_runtime_adapter import (
    AgentRoutingToolRuntimeAdapter,
)
from design_research_agents.agent.internal.input_parsing import (
    extract_prompt as _extract_prompt,
)
from design_research_agents.agent.internal.prompt_overrides import validate_prompt_text
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.contracts.workflow import LogicStep
from design_research_agents.tracing import Tracer, finish_trace_run, start_trace_run
from design_research_agents.workflow.implementations.workflow_runtime import WorkflowRuntime
from design_research_agents.workflow.internal import (
    WorkflowBudgetTracker,
    attach_runtime_metadata,
    build_pattern_failure_result,
    merge_dependencies,
    normalize_request_id_prefix,
    resolve_request_id_with_prefix,
)


class RouterPattern(Agent):
    """Routing/delegation pattern built on workflow primitives."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        alternatives: Mapping[str, Agent],
        alternative_descriptions: Mapping[str, str] | None = None,
        controls: RuntimeControls | None = None,
        agent_routing_router_system_prompt: str | None = None,
        agent_routing_router_user_prompt_template: str | None = None,
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and initialize workflow-native routing settings.

        Args:
            llm_client: LLM client used by the router agent.
            tool_runtime: Tool runtime used to cost/metadata-account delegated calls.
            alternatives: Mapping of route keys to delegate agents.
            alternative_descriptions: Optional descriptions used to guide routing.
            controls: Runtime controls (budgets, streaming, limits).
            agent_routing_router_system_prompt: Optional override for router system prompt.
            agent_routing_router_user_prompt_template: Optional override for router user prompt.
            default_request_id_prefix: Optional prefix used to derive request ids.
            default_dependencies: Dependency defaults merged into each run.
            tracer: Optional tracer used for run-level instrumentation.

        Raises:
            ValueError: If no valid route alternatives are supplied.
        """
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._controls = controls or RuntimeControls()
        self._tracer = tracer
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
                value=agent_routing_router_system_prompt,
                field_name="agent_routing_router_system_prompt",
            )
            if agent_routing_router_system_prompt is not None
            else None
        )
        self._router_user_prompt_template = (
            validate_prompt_text(
                value=agent_routing_router_user_prompt_template,
                field_name="agent_routing_router_user_prompt_template",
            )
            if agent_routing_router_user_prompt_template is not None
            else None
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
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
        configured_request_id = resolve_request_id_with_prefix(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        resolved_request_id = resolve_request_id(configured_request_id)
        resolved_dependencies = normalize_dependencies(
            merge_dependencies(
                default_dependencies=self._default_dependencies,
                run_dependencies=dependencies,
            )
        )
        normalized_input = normalize_input_payload(prompt)
        resolved_prompt = _extract_prompt(normalized_input)
        trace_scope = start_trace_run(
            agent_name="RouterPattern",
            request_id=resolved_request_id,
            input_payload={"prompt": resolved_prompt, "mode": "agent_routing"},
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )

        try:
            result = self._run_agent_routing(
                prompt=resolved_prompt,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
            )
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise

        finish_trace_run(trace_scope, result=result)
        return result

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Execute one run and emit wrapper-style stream events.

        Args:
            prompt: User prompt to route to the best delegate.
            request_id: Optional request id for tracing and correlation.
            dependencies: Optional dependency overrides for this run.

        Yields:
            Delta and completed events derived from ``run`` output.
        """
        runtime_result = self.run(prompt, request_id=request_id, dependencies=dependencies)
        if self._controls.streaming_enabled:
            delta_text = (
                runtime_result.model_response.text
                if runtime_result.model_response is not None
                else ""
            )
            yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=runtime_result)

    def _run_agent_routing(  # noqa: C901
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
    ) -> AgentResult:
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
        router_agent = SingleStepRouterAgent(
            llm_client=self._llm_client,
            tool_runtime=routing_tool_runtime,
            system_prompt=self._router_system_prompt,
            user_prompt_template=self._router_user_prompt_template,
            tracer=self._tracer,
        )
        workflow_runtime = WorkflowRuntime(tracer=self._tracer)

        runtime_tool_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        router_result: AgentResult | None = None
        delegated_result: AgentResult | None = None

        def _run_selection(context: Mapping[str, object]) -> Mapping[str, object]:
            """Execute router-model selection step.

            Args:
                context: Step execution context (unused by selection logic).

            Returns:
                Selection status payload consumed by downstream delegate step.
            """
            del context
            nonlocal router_result
            router_result = router_agent.run(
                prompt,
                request_id=f"{request_id}:agent_routing_router",
                dependencies=dependencies,
            )
            budget_tracker.add_model_response(router_result.model_response)
            if not router_result.success:
                return {
                    "status": "routing_failure",
                    "routing": router_result.metadata.get("routing", {}),
                }

            selected_name = str(router_result.output.get("tool_name", "")).strip()
            return {
                "status": "selected",
                "selected_name": selected_name,
                "routing": router_result.metadata.get("routing", {}),
            }

        def _run_delegate(context: Mapping[str, object]) -> Mapping[str, object]:
            """Execute selected delegate agent based on selection-step output.

            Args:
                context: Step execution context containing dependency step outputs.

            Returns:
                Delegate status payload and selected route metadata.
            """
            nonlocal delegated_result
            dependency_results = context.get("dependency_results")
            if not isinstance(dependency_results, Mapping):
                return {
                    "status": "routing_failure",
                    "routing": {},
                }
            selection_step = dependency_results.get("agent_routing_selection")
            if not isinstance(selection_step, Mapping):
                return {
                    "status": "routing_failure",
                    "routing": {},
                }
            selection_output = selection_step.get("output")
            if not isinstance(selection_output, Mapping):
                return {
                    "status": "routing_failure",
                    "routing": {},
                }

            status = selection_output.get("status")
            if status != "selected":
                return {
                    "status": "routing_failure",
                    "routing": selection_output.get("routing", {}),
                }

            selected_name = str(selection_output.get("selected_name", "")).strip()
            selected_agent = self._alternatives.get(selected_name)
            if selected_agent is None:
                return {
                    "status": "unknown_alternative",
                    "selected_name": selected_name,
                    "routing": selection_output.get("routing", {}),
                }

            delegated_result = selected_agent.run(
                prompt,
                request_id=f"{request_id}:agent_routing:{selected_name}",
                dependencies=dependencies,
            )
            budget_tracker.add_model_response(delegated_result.model_response)
            budget_tracker.add_tool_results(
                tool_results=delegated_result.tool_results,
                tool_specs=runtime_tool_specs,
            )
            return {
                "status": "delegated",
                "selected_name": selected_name,
                "delegated_success": delegated_result.success,
                "routing": selection_output.get("routing", {}),
            }

        workflow_result = workflow_runtime.run(
            steps=[
                LogicStep(step_id="agent_routing_selection", handler=_run_selection),
                LogicStep(
                    step_id="agent_routing_delegate",
                    dependencies=("agent_routing_selection",),
                    handler=_run_delegate,
                ),
            ],
            context={"prompt": prompt},
            execution_mode="sequential",
            failure_policy="skip_dependents",
            request_id=f"{request_id}:agent_routing_workflow",
            dependencies=dependencies,
        )

        if not workflow_result.success:
            raise RuntimeError("Agent routing workflow graph execution failed.")

        if router_result is None:
            raise RuntimeError("Agent routing selection step did not produce a router result.")

        selection_step = workflow_result.step_results.get("agent_routing_selection")
        selection_output = selection_step.output if selection_step is not None else {}
        delegate_step = workflow_result.step_results.get("agent_routing_delegate")
        delegate_output = delegate_step.output if delegate_step is not None else {}

        if not router_result.success:
            failure = build_pattern_failure_result(
                error="Agent routing selection failed.",
                model_response=router_result.model_response,
                request_id=request_id,
                dependencies=dependencies,
                metadata={
                    "stage": "agent_routing_selection",
                    "mode": "agent_routing",
                    "routing": router_result.metadata.get("routing", {}),
                },
                output={
                    "terminated_reason": "routing_failure",
                    "routing": router_result.metadata.get("routing", {}),
                    "delegated_agent": None,
                    "delegated_output": {},
                },
            )
            return attach_runtime_metadata(
                agent_result=failure,
                requested_mode="agent_routing",
                resolved_mode="agent_routing",
                controls=self._controls,
                budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                extra_metadata=None,
            )

        selected_name = str(selection_output.get("selected_name", "")).strip()
        if (
            delegate_output.get("status") == "unknown_alternative"
            or selected_name not in self._alternatives
        ):
            failure = build_pattern_failure_result(
                error=f"Agent routing selected unknown agent alternative '{selected_name}'.",
                model_response=router_result.model_response,
                request_id=request_id,
                dependencies=dependencies,
                metadata={
                    "stage": "agent_routing_selection",
                    "mode": "agent_routing",
                    "routing": router_result.metadata.get("routing", {}),
                },
                output={
                    "terminated_reason": "unknown_alternative",
                    "routing": router_result.metadata.get("routing", {}),
                    "delegated_agent": None,
                    "delegated_output": {},
                },
            )
            return attach_runtime_metadata(
                agent_result=failure,
                requested_mode="agent_routing",
                resolved_mode="agent_routing",
                controls=self._controls,
                budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                extra_metadata=None,
            )

        if delegated_result is None:
            failure = build_pattern_failure_result(
                error="Agent routing delegate execution did not run.",
                model_response=router_result.model_response,
                request_id=request_id,
                dependencies=dependencies,
                metadata={
                    "stage": "agent_routing_delegate",
                    "mode": "agent_routing",
                    "routing": router_result.metadata.get("routing", {}),
                },
                output={
                    "terminated_reason": "routing_failure",
                    "routing": router_result.metadata.get("routing", {}),
                    "delegated_agent": None,
                    "delegated_output": {},
                },
            )
            return attach_runtime_metadata(
                agent_result=failure,
                requested_mode="agent_routing",
                resolved_mode="agent_routing",
                controls=self._controls,
                budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                extra_metadata=None,
            )

        agent_routing_metadata = {
            "routing": router_result.metadata.get("routing", {}),
            "selected_alternative": selected_name,
            "available_alternatives": sorted(self._alternatives.keys()),
        }

        delegated_output = dict(delegated_result.output)
        delegated_output["agent_routing_selected_alternative"] = selected_name

        result = AgentResult(
            output=delegated_output,
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
            controls=self._controls,
            budget_metadata=budget_tracker.as_metadata(controls=self._controls),
            extra_metadata={
                "workflow": {
                    "execution_order": list(workflow_result.execution_order),
                }
            },
        )


__all__ = [
    "RouterPattern",
]
