"""Unified multi-mode agent runtime.

``AgentRuntime`` exposes one entrypoint that can execute different multi-agent
patterns while reusing existing concrete agents in this package.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from design_research_agents.agent.implementations.multi_step_code_tool_calling_agent import (
    MultiStepCodeToolCallingAgent,
)
from design_research_agents.agent.implementations.single_step_code_tool_calling_agent import (
    SingleStepCodeToolCallingAgent,
)
from design_research_agents.agent.implementations.single_step_direct_llm_agent import (
    SingleStepDirectLLMAgent,
)
from design_research_agents.agent.implementations.single_step_router_agent import (
    SingleStepRouterAgent,
)
from design_research_agents.agent.internal.agent_routing_runtime_adapter import (
    AgentRoutingToolRuntimeAdapter,
)
from design_research_agents.agent.internal.input_parsing import (
    extract_prompt as _extract_prompt,
)
from design_research_agents.agent.internal.input_parsing import (
    parse_json_mapping as _parse_json_mapping,
)
from design_research_agents.agent.internal.model_resolution import resolve_agent_model
from design_research_agents.agent.internal.result_builders import build_failure_result
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import (
    LLMChatParams,
    LLMClient,
    LLMMessage,
    LLMResponse,
)
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.schemas import (
    SchemaValidationError,
    validate_payload_against_schema,
)
from design_research_agents.tracing import (
    Tracer,
    finish_model_call,
    finish_trace_run,
    start_model_call,
    start_trace_run,
)

RuntimeMode = Literal["react", "plan_execute", "propose_critic", "agent_routing"]

_PLAN_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps"],
    "properties": {
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["step_id", "instruction", "success_criteria"],
                "properties": {
                    "step_id": {"type": "string"},
                    "instruction": {"type": "string"},
                    "success_criteria": {"type": "string"},
                },
            },
        }
    },
}

_CRITIC_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["approved", "feedback", "revision_goals"],
    "properties": {
        "approved": {"type": "boolean"},
        "feedback": {"type": "string"},
        "revision_goals": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
}


@dataclass(slots=True)
class _BudgetTracker:
    """Soft budget accumulator used for runtime metadata."""

    observed_latency_ms: int = 0
    observed_model_calls: int = 0
    observed_tool_calls: int = 0
    observed_estimated_usd: float = 0.0

    def add_model_response(self, model_response: LLMResponse | None) -> None:
        """Accumulate model-call latency metrics from one response."""
        if model_response is None:
            return
        self.observed_model_calls += 1
        if isinstance(model_response.latency_ms, int) and model_response.latency_ms >= 0:
            self.observed_latency_ms += model_response.latency_ms

    def add_tool_results(
        self,
        *,
        tool_results: list[ToolResult],
        tool_specs: Mapping[str, ToolSpec],
    ) -> None:
        """Accumulate tool-call counts and estimated USD cost."""
        for tool_result in tool_results:
            self.observed_tool_calls += 1
            runtime_spec = tool_specs.get(tool_result.tool_name)
            if runtime_spec is None:
                continue
            estimated_cost = runtime_spec.cost_hints.usd_cost_estimate
            if isinstance(estimated_cost, (int, float)):
                self.observed_estimated_usd += float(estimated_cost)

    def as_metadata(self, *, controls: RuntimeControls) -> dict[str, object]:
        """Return soft-budget metadata with exceeded flags."""
        latency_exceeded = (
            controls.soft_max_latency_ms is not None
            and self.observed_latency_ms > controls.soft_max_latency_ms
        )
        usd_exceeded = (
            controls.soft_max_usd is not None
            and self.observed_estimated_usd > controls.soft_max_usd
        )
        return {
            "observed_latency_ms": self.observed_latency_ms,
            "observed_model_calls": self.observed_model_calls,
            "observed_tool_calls": self.observed_tool_calls,
            "observed_estimated_usd": round(self.observed_estimated_usd, 6),
            "soft_max_latency_ms": controls.soft_max_latency_ms,
            "soft_max_usd": controls.soft_max_usd,
            "latency_exceeded": latency_exceeded,
            "usd_exceeded": usd_exceeded,
        }


class AgentRuntime(Agent):
    """Unified runtime that exposes multiple execution patterns via ``mode``."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        mode: RuntimeMode,
        controls: RuntimeControls | None = None,
        agent_routing_alternatives: Mapping[str, Agent] | None = None,
        agent_routing_descriptions: Mapping[str, str] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a multi-mode runtime.

        Args:
            llm_client: LLM client used for model-backed phases.
            tool_runtime: Runtime used for tool-enabled modes.
            mode: Active execution mode.
            controls: Shared runtime controls.
            agent_routing_alternatives: Constructor-provided agent-routing alternatives.
            agent_routing_descriptions: Optional alternative descriptions for agent routing.
            tracer: Optional explicit tracer dependency.
        """
        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._mode = mode
        self._tracer = tracer
        self._controls = controls or RuntimeControls()
        self._agent_routing_alternatives = {
            name.strip(): agent
            for name, agent in (agent_routing_alternatives or {}).items()
            if isinstance(name, str) and name.strip()
        }
        self._agent_routing_descriptions = {
            name.strip(): description.strip()
            for name, description in (agent_routing_descriptions or {}).items()
            if isinstance(name, str)
            and name.strip()
            and isinstance(description, str)
            and description.strip()
        }

        if self._mode == "agent_routing" and not self._agent_routing_alternatives:
            raise ValueError("agent_routing_alternatives must be given for mode='agent_routing'.")

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one run using the configured runtime mode."""
        resolved_request_id = resolve_request_id(request_id)
        resolved_dependencies = normalize_dependencies(dependencies)
        normalized_input = normalize_input_payload(prompt)
        resolved_prompt = _extract_prompt(normalized_input)
        trace_scope = start_trace_run(
            agent_name="AgentRuntime",
            request_id=resolved_request_id,
            input_payload={"prompt": resolved_prompt, "mode": self._mode},
            dependencies=resolved_dependencies,
            tracer=self._tracer,
        )

        try:
            mode_result = self._run_mode(
                prompt=resolved_prompt,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
                normalized_input=normalized_input,
            )
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise
        finish_trace_run(trace_scope, result=mode_result)
        return mode_result

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Run one mode execution and emit stream events."""
        if self._mode == "react":
            react_agent = self._build_react_agent()
            for event in react_agent.run_stream(
                prompt,
                request_id=request_id,
                dependencies=dependencies,
            ):
                if event.kind != "completed" or event.result is None:
                    if self._controls.streaming_enabled:
                        yield event
                    continue
                yield AgentStreamEvent(
                    kind="completed",
                    result=self._attach_runtime_metadata(
                        agent_result=event.result,
                        requested_mode="react",
                        resolved_mode="multi_step_code_tool_calling_agent",
                        budget_metadata=_budget_for_result(
                            agent_result=event.result,
                            controls=self._controls,
                            tool_runtime=self._tool_runtime,
                        ),
                        extra_metadata=None,
                    ),
                )
            return

        runtime_result = self.run(prompt, request_id=request_id, dependencies=dependencies)
        if self._controls.streaming_enabled:
            delta_text = (
                runtime_result.model_response.text
                if runtime_result.model_response is not None
                else ""
            )
            yield AgentStreamEvent(kind="delta", delta_text=delta_text)
        yield AgentStreamEvent(kind="completed", result=runtime_result)

    def _run_mode(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        normalized_input: Mapping[str, object],
    ) -> AgentResult:
        if self._mode == "react":
            react_result = self._build_react_agent().run(
                prompt,
                request_id=request_id,
                dependencies=dependencies,
            )
            return self._attach_runtime_metadata(
                agent_result=react_result,
                requested_mode="react",
                resolved_mode="multi_step_code_tool_calling_agent",
                budget_metadata=_budget_for_result(
                    agent_result=react_result,
                    controls=self._controls,
                    tool_runtime=self._tool_runtime,
                ),
                extra_metadata=None,
            )
        if self._mode == "plan_execute":
            return self._run_plan_execute(
                prompt=prompt,
                request_id=request_id,
                dependencies=dependencies,
                normalized_input=normalized_input,
            )
        if self._mode == "propose_critic":
            return self._run_propose_critic(
                prompt=prompt,
                request_id=request_id,
                dependencies=dependencies,
                normalized_input=normalized_input,
            )
        if self._mode == "agent_routing":
            return self._run_agent_routing(
                prompt=prompt,
                request_id=request_id,
                dependencies=dependencies,
                normalized_input=normalized_input,
            )
        raise ValueError(f"Unsupported runtime mode '{self._mode}'.")

    def _run_plan_execute(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        normalized_input: Mapping[str, object],
    ) -> AgentResult:
        budget_tracker = _BudgetTracker()
        runtime_tool_specs = {spec.name: spec for spec in self._tool_runtime.list_tools()}
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )

        planner_messages = [
            LLMMessage(
                role="system",
                content=(
                    "You are a planner for a plan-execute runtime. "
                    "Return strict JSON only with steps[]."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    "Create an execution plan for this task. "
                    "Each step must have step_id, instruction, and success_criteria.\n\n"
                    f"Task:\n{prompt}"
                ),
            ),
        ]
        planner_params = LLMChatParams(
            response_schema=dict(_PLAN_SCHEMA),
            provider_options={
                "agent": "AgentRuntime",
                "mode": "plan_execute",
                "phase": "planner",
            },
        )
        planner_span_id = start_model_call(
            model=resolved_model,
            messages=planner_messages,
            params=planner_params,
            metadata={
                "agent": "AgentRuntime",
                "mode": "plan_execute",
                "phase": "planner",
            },
        )
        try:
            planner_response = self._llm_client.chat(
                planner_messages,
                model=resolved_model,
                params=planner_params,
            )
        except Exception as exc:
            finish_model_call(planner_span_id, error=str(exc), model=resolved_model)
            raise
        finish_model_call(planner_span_id, response=planner_response)
        budget_tracker.add_model_response(planner_response)

        parsed_plan = _parse_json_mapping(planner_response.text)
        if parsed_plan is None:
            failure = _failure_result(
                error="Planner did not return valid JSON plan output.",
                model_response=planner_response,
                request_id=request_id,
                dependencies=dependencies,
                metadata={"stage": "planner", "mode": "plan_execute"},
                output={
                    "terminated_reason": "planner_invalid_json",
                    "plan": None,
                    "steps_executed": 0,
                    "step_results": [],
                    "final_output": {},
                },
            )
            return self._attach_runtime_metadata(
                agent_result=failure,
                requested_mode="plan_execute",
                resolved_mode="plan_execute",
                budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                extra_metadata=None,
            )

        try:
            validate_payload_against_schema(
                payload=parsed_plan,
                schema=_PLAN_SCHEMA,
                location="plan_execute.plan",
            )
        except SchemaValidationError as exc:
            failure = _failure_result(
                error=f"Planner output failed schema validation: {exc}",
                model_response=planner_response,
                request_id=request_id,
                dependencies=dependencies,
                metadata={"stage": "planner", "mode": "plan_execute"},
                output={
                    "terminated_reason": "planner_invalid_schema",
                    "plan": parsed_plan,
                    "steps_executed": 0,
                    "step_results": [],
                    "final_output": {},
                },
            )
            return self._attach_runtime_metadata(
                agent_result=failure,
                requested_mode="plan_execute",
                resolved_mode="plan_execute",
                budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                extra_metadata=None,
            )

        raw_steps = parsed_plan.get("steps")
        plan_steps = raw_steps if isinstance(raw_steps, list) else []

        executor_agent = SingleStepCodeToolCallingAgent(
            llm_client=self._llm_client,
            tool_runtime=self._tool_runtime,
            max_tool_calls=self._controls.max_tool_calls_per_step,
            execution_timeout_seconds=self._controls.execution_timeout_seconds_per_step,
            tracer=self._tracer,
        )

        step_results: list[dict[str, object]] = []
        all_tool_results: list[ToolResult] = []
        final_output: dict[str, object] = {}
        terminated_reason = "completed"
        last_model_response: LLMResponse | None = planner_response

        execution_limit = min(len(plan_steps), self._controls.max_iterations)
        for index in range(execution_limit):
            raw_step = plan_steps[index]
            if not isinstance(raw_step, Mapping):
                continue
            step_id = str(raw_step.get("step_id", f"step_{index + 1}"))
            step_instruction = str(raw_step.get("instruction", ""))
            success_criteria = str(raw_step.get("success_criteria", ""))
            step_prompt = "\n".join(
                [
                    f"Task: {prompt}",
                    f"Plan step id: {step_id}",
                    f"Instruction: {step_instruction}",
                    f"Success criteria: {success_criteria}",
                    "Prior step outputs:",
                    json.dumps(step_results[-3:], sort_keys=True),
                ]
            )

            step_result = executor_agent.run(
                step_prompt,
                request_id=f"{request_id}:plan-step-{index + 1}",
                dependencies=dependencies,
            )
            budget_tracker.add_model_response(step_result.model_response)
            budget_tracker.add_tool_results(
                tool_results=step_result.tool_results,
                tool_specs=runtime_tool_specs,
            )
            if step_result.model_response is not None:
                last_model_response = step_result.model_response
            all_tool_results.extend(step_result.tool_results)

            step_record = {
                "step_id": step_id,
                "instruction": step_instruction,
                "success_criteria": success_criteria,
                "success": step_result.success,
                "final_output": step_result.output.get("final_output", {}),
                "error": step_result.output.get("error"),
            }
            step_results.append(step_record)

            if step_result.success:
                maybe_output = step_result.output.get("final_output")
                if isinstance(maybe_output, Mapping):
                    final_output = dict(maybe_output)
                continue

            terminated_reason = "step_failure"
            break

        if terminated_reason != "step_failure" and len(plan_steps) > self._controls.max_iterations:
            terminated_reason = "max_iterations_reached"

        success = terminated_reason in {"completed", "max_iterations_reached"} and bool(
            step_results
        )
        plan_execute_result = AgentResult(
            output={
                "plan": parsed_plan,
                "steps_executed": len(step_results),
                "step_results": step_results,
                "final_output": final_output,
                "terminated_reason": terminated_reason,
            },
            success=success,
            tool_results=all_tool_results,
            model_response=last_model_response,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "stage": "execution",
                "mode": "plan_execute",
            },
        )
        return self._attach_runtime_metadata(
            agent_result=plan_execute_result,
            requested_mode="plan_execute",
            resolved_mode="plan_execute",
            budget_metadata=budget_tracker.as_metadata(controls=self._controls),
            extra_metadata={
                "plan": {
                    "step_count": len(plan_steps),
                    "executed_step_count": len(step_results),
                },
            },
        )

    def _run_propose_critic(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        normalized_input: Mapping[str, object],
    ) -> AgentResult:
        budget_tracker = _BudgetTracker()
        resolved_model = resolve_agent_model(
            llm_client=self._llm_client,
        )
        proposer = SingleStepDirectLLMAgent(
            llm_client=self._llm_client,
            default_system_prompt=(
                "You are a proposer. Produce a concrete draft response for the task."
            ),
            tracer=self._tracer,
        )

        critique_iterations: list[dict[str, object]] = []
        current_feedback = ""
        current_goals: list[str] = []
        current_proposal = ""
        last_model_response: LLMResponse | None = None
        terminated_reason = "max_iterations_reached"
        approved = False

        for iteration in range(self._controls.max_iterations):
            propose_prompt = "\n".join(
                [
                    f"Task: {prompt}",
                    f"Iteration: {iteration + 1}",
                    f"Prior feedback: {current_feedback or '(none)'}",
                    f"Revision goals: {json.dumps(current_goals, sort_keys=True)}",
                    "Return only the revised proposal text.",
                ]
            )
            propose_result = proposer.run(
                propose_prompt,
                request_id=f"{request_id}:propose-{iteration + 1}",
                dependencies=dependencies,
            )
            if propose_result.model_response is not None:
                last_model_response = propose_result.model_response
                budget_tracker.add_model_response(propose_result.model_response)
            current_proposal = str(propose_result.output.get("model_text", "")).strip()

            critic_messages = [
                LLMMessage(
                    role="system",
                    content=(
                        "You are a strict critic. Return JSON only with approved, feedback, "
                        "revision_goals."
                    ),
                ),
                LLMMessage(
                    role="user",
                    content=(
                        f"Task:\n{prompt}\n\n"
                        f"Proposal:\n{current_proposal}\n\n"
                        "Critique and return structured JSON."
                    ),
                ),
            ]
            critic_params = LLMChatParams(
                response_schema=dict(_CRITIC_SCHEMA),
                provider_options={
                    "agent": "AgentRuntime",
                    "mode": "propose_critic",
                    "phase": "critic",
                },
            )
            critic_span_id = start_model_call(
                model=resolved_model,
                messages=critic_messages,
                params=critic_params,
                metadata={
                    "agent": "AgentRuntime",
                    "mode": "propose_critic",
                    "phase": "critic",
                },
            )
            try:
                critic_response = self._llm_client.chat(
                    critic_messages,
                    model=resolved_model,
                    params=critic_params,
                )
            except Exception as exc:
                finish_model_call(critic_span_id, error=str(exc), model=resolved_model)
                raise
            finish_model_call(critic_span_id, response=critic_response)
            last_model_response = critic_response
            budget_tracker.add_model_response(critic_response)

            parsed_critique = _parse_json_mapping(critic_response.text)
            if parsed_critique is None:
                failure = _failure_result(
                    error="Critic did not return valid JSON output.",
                    model_response=critic_response,
                    request_id=request_id,
                    dependencies=dependencies,
                    metadata={"stage": "critic", "mode": "propose_critic"},
                    output={
                        "proposal": current_proposal,
                        "critique_iterations": critique_iterations,
                        "terminated_reason": "critic_invalid_json",
                    },
                )
                return self._attach_runtime_metadata(
                    agent_result=failure,
                    requested_mode="propose_critic",
                    resolved_mode="propose_critic",
                    budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                    extra_metadata=None,
                )

            try:
                validate_payload_against_schema(
                    payload=parsed_critique,
                    schema=_CRITIC_SCHEMA,
                    location="propose_critic.critic",
                )
            except SchemaValidationError as exc:
                failure = _failure_result(
                    error=f"Critic output failed schema validation: {exc}",
                    model_response=critic_response,
                    request_id=request_id,
                    dependencies=dependencies,
                    metadata={"stage": "critic", "mode": "propose_critic"},
                    output={
                        "proposal": current_proposal,
                        "critique_iterations": critique_iterations,
                        "terminated_reason": "critic_invalid_schema",
                    },
                )
                return self._attach_runtime_metadata(
                    agent_result=failure,
                    requested_mode="propose_critic",
                    resolved_mode="propose_critic",
                    budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                    extra_metadata=None,
                )

            approved = bool(parsed_critique.get("approved"))
            feedback = str(parsed_critique.get("feedback", ""))
            revision_goals_raw = parsed_critique.get("revision_goals")
            revision_goals = (
                [str(goal) for goal in revision_goals_raw]
                if isinstance(revision_goals_raw, list)
                else []
            )
            critique_iterations.append(
                {
                    "iteration": iteration + 1,
                    "proposal": current_proposal,
                    "approved": approved,
                    "feedback": feedback,
                    "revision_goals": revision_goals,
                }
            )

            if approved:
                terminated_reason = "approved"
                break

            current_feedback = feedback
            current_goals = revision_goals

        success = approved
        propose_critic_result = AgentResult(
            output={
                "proposal": current_proposal,
                "critique_iterations": critique_iterations,
                "terminated_reason": terminated_reason,
                "approved": approved,
            },
            success=success,
            tool_results=[],
            model_response=last_model_response,
            metadata={
                "request_id": request_id,
                "dependency_keys": sorted(dependencies.keys()),
                "mode": "propose_critic",
                "iterations": len(critique_iterations),
            },
        )
        return self._attach_runtime_metadata(
            agent_result=propose_critic_result,
            requested_mode="propose_critic",
            resolved_mode="propose_critic",
            budget_metadata=budget_tracker.as_metadata(controls=self._controls),
            extra_metadata=None,
        )

    def _run_agent_routing(
        self,
        *,
        prompt: str,
        request_id: str,
        dependencies: Mapping[str, object],
        normalized_input: Mapping[str, object],
    ) -> AgentResult:
        budget_tracker = _BudgetTracker()
        agent_routing_tool_runtime = AgentRoutingToolRuntimeAdapter(
            alternatives=self._agent_routing_alternatives,
            descriptions=self._agent_routing_descriptions,
        )
        single_step_router_agent = SingleStepRouterAgent(
            llm_client=self._llm_client,
            tool_runtime=agent_routing_tool_runtime,
            tracer=self._tracer,
        )
        router_result = single_step_router_agent.run(
            prompt,
            request_id=f"{request_id}:agent_routing_router",
            dependencies=dependencies,
        )
        budget_tracker.add_model_response(router_result.model_response)

        if not router_result.success:
            failure = _failure_result(
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
            return self._attach_runtime_metadata(
                agent_result=failure,
                requested_mode="agent_routing",
                resolved_mode="agent_routing",
                budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                extra_metadata=None,
            )

        selected_name = str(router_result.output.get("tool_name", "")).strip()
        selected_agent = self._agent_routing_alternatives.get(selected_name)
        if selected_agent is None:
            failure = _failure_result(
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
            return self._attach_runtime_metadata(
                agent_result=failure,
                requested_mode="agent_routing",
                resolved_mode="agent_routing",
                budget_metadata=budget_tracker.as_metadata(controls=self._controls),
                extra_metadata=None,
            )

        delegated_result = selected_agent.run(
            prompt,
            request_id=f"{request_id}:agent_routing:{selected_name}",
            dependencies=dependencies,
        )
        budget_tracker.add_model_response(delegated_result.model_response)
        budget_tracker.add_tool_results(
            tool_results=delegated_result.tool_results,
            tool_specs={spec.name: spec for spec in self._tool_runtime.list_tools()},
        )

        agent_routing_metadata = {
            "routing": router_result.metadata.get("routing", {}),
            "selected_alternative": selected_name,
            "available_alternatives": sorted(self._agent_routing_alternatives.keys()),
        }

        delegated_output = dict(delegated_result.output)
        delegated_output["agent_routing_selected_alternative"] = selected_name

        agent_routing_result = AgentResult(
            output=delegated_output,
            success=delegated_result.success,
            tool_results=list(delegated_result.tool_results),
            model_response=delegated_result.model_response,
            metadata={
                **dict(delegated_result.metadata),
                "agent_routing": agent_routing_metadata,
            },
        )
        return self._attach_runtime_metadata(
            agent_result=agent_routing_result,
            requested_mode="agent_routing",
            resolved_mode="agent_routing",
            budget_metadata=budget_tracker.as_metadata(controls=self._controls),
            extra_metadata=None,
        )

    def _build_react_agent(self) -> MultiStepCodeToolCallingAgent:
        """Construct the delegated ``MultiStepCodeToolCallingAgent`` for react mode."""
        return MultiStepCodeToolCallingAgent(
            llm_client=self._llm_client,
            tool_runtime=self._tool_runtime,
            max_steps=self._controls.max_steps,
            max_tool_calls_per_step=self._controls.max_tool_calls_per_step,
            execution_timeout_seconds_per_step=self._controls.execution_timeout_seconds_per_step,
            tracer=self._tracer,
        )

    def _attach_runtime_metadata(
        self,
        *,
        agent_result: AgentResult,
        requested_mode: RuntimeMode,
        resolved_mode: str,
        budget_metadata: Mapping[str, object],
        extra_metadata: Mapping[str, object] | None,
    ) -> AgentResult:
        metadata = dict(agent_result.metadata)
        runtime_metadata: dict[str, object] = {
            "requested_mode": requested_mode,
            "resolved_mode": resolved_mode,
            "controls": self._controls.asdict(),
            "soft_budget": dict(budget_metadata),
        }
        if extra_metadata is not None:
            runtime_metadata.update(extra_metadata)
        metadata["runtime"] = runtime_metadata
        return AgentResult(
            output=dict(agent_result.output),
            success=agent_result.success,
            tool_results=list(agent_result.tool_results),
            model_response=agent_result.model_response,
            metadata=metadata,
        )


def _failure_result(
    *,
    error: str,
    model_response: LLMResponse | None,
    request_id: str,
    dependencies: Mapping[str, object],
    metadata: Mapping[str, object],
    output: Mapping[str, object],
) -> AgentResult:
    return build_failure_result(
        error=error,
        model_response=model_response,
        tool_results=[],
        request_id=request_id,
        dependencies=dependencies,
        metadata=metadata,
        output=output,
    )


def _budget_for_result(
    *,
    agent_result: AgentResult,
    controls: RuntimeControls,
    tool_runtime: ToolRuntime,
) -> dict[str, object]:
    tracker = _BudgetTracker()
    tracker.add_model_response(agent_result.model_response)
    tracker.add_tool_results(
        tool_results=agent_result.tool_results,
        tool_specs={spec.name: spec for spec in tool_runtime.list_tools()},
    )
    return tracker.as_metadata(controls=controls)
