"""Unified react-only agent runtime."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal

from design_research_agents.agent.implementations.multi_step_code_tool_calling_agent import (
    MultiStepCodeToolCallingAgent,
)
from design_research_agents.agent.internal.input_parsing import (
    extract_prompt as _extract_prompt,
)
from design_research_agents.agent.internal.run_options import (
    normalize_dependencies,
    normalize_input_payload,
    resolve_request_id,
)
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.tools import ToolResult, ToolRuntime, ToolSpec
from design_research_agents.tracing import Tracer, finish_trace_run, start_trace_run

RuntimeMode = Literal["react"]


@dataclass(slots=True)
class _BudgetTracker:
    """Soft budget accumulator used for runtime metadata."""

    observed_latency_ms: int = 0
    """Field value for ``observed_latency_ms``."""
    observed_model_calls: int = 0
    """Field value for ``observed_model_calls``."""
    observed_tool_calls: int = 0
    """Field value for ``observed_tool_calls``."""
    observed_estimated_usd: float = 0.0
    """Field value for ``observed_estimated_usd``."""

    def add_model_response(self, model_response: object | None) -> None:
        """Accumulate model-call latency metrics from one optional response.

        Args:
            model_response: Parameter value.
        """
        if model_response is None:
            return
        self.observed_model_calls += 1
        latency_ms = getattr(model_response, "latency_ms", None)
        if isinstance(latency_ms, int) and latency_ms >= 0:
            self.observed_latency_ms += latency_ms

    def add_tool_results(
        self,
        *,
        tool_results: list[ToolResult],
        tool_specs: Mapping[str, ToolSpec],
    ) -> None:
        """Accumulate tool-call counts and estimated USD cost.

        Args:
            tool_results: Parameter value.
            tool_specs: Parameter value.
        """
        for tool_result in tool_results:
            self.observed_tool_calls += 1
            runtime_spec = tool_specs.get(tool_result.tool_name)
            if runtime_spec is None:
                continue
            estimated_cost = runtime_spec.cost_hints.usd_cost_estimate
            if isinstance(estimated_cost, (int, float)):
                self.observed_estimated_usd += float(estimated_cost)

    def as_metadata(self, *, controls: RuntimeControls) -> dict[str, object]:
        """Return soft-budget metadata with exceeded flags.

        Args:
            controls: Parameter value.

        Returns:
            The resulting value.
        """
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
    """React-only runtime that delegates to ``MultiStepCodeToolCallingAgent``."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        mode: str = "react",
        controls: RuntimeControls | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Initialize a react-only runtime.

        Args:
            llm_client: LLM client used by the delegated react agent.
            tool_runtime: Runtime used for tool invocation.
            mode: Runtime mode. Only ``"react"`` is supported.
            controls: Shared runtime controls.
            tracer: Optional explicit tracer dependency.

        Raises:
            Exception: Raised when execution fails.
        """
        if mode != "react":
            raise ValueError(
                "AgentRuntime now supports mode='react' only. "
                "Use PlannerExecutorPattern, ReflexionPattern, or "
                "RouterPattern for multi-agent orchestration."
            )

        self._llm_client = llm_client
        self._tool_runtime = tool_runtime
        self._mode: RuntimeMode = "react"
        self._controls = controls or RuntimeControls()
        self._tracer = tracer

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one react-mode run and return the final result.

        Args:
            prompt: Parameter value.
            request_id: Parameter value.
            dependencies: Parameter value.

        Returns:
            The resulting value.

        Raises:
            Exception: Raised when execution fails.
        """
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
            react_result = self._build_react_agent().run(
                resolved_prompt,
                request_id=resolved_request_id,
                dependencies=resolved_dependencies,
            )
        except Exception as exc:
            finish_trace_run(trace_scope, error=str(exc))
            raise

        runtime_result = self._attach_runtime_metadata(
            agent_result=react_result,
            requested_mode="react",
            resolved_mode="multi_step_code_tool_calling_agent",
            budget_metadata=_budget_for_result(
                agent_result=react_result,
                controls=self._controls,
                tool_runtime=self._tool_runtime,
            ),
        )
        finish_trace_run(trace_scope, result=runtime_result)
        return runtime_result

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Run one react-mode execution and emit stream events.

        Args:
            prompt: Parameter value.
            request_id: Parameter value.
            dependencies: Parameter value.

        Yields:
            The yielded values.
        """
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
                ),
            )

    def _build_react_agent(self) -> MultiStepCodeToolCallingAgent:
        """Construct delegated ``MultiStepCodeToolCallingAgent`` for react mode.

        Returns:
            The resulting value.
        """
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
    ) -> AgentResult:
        """Run attach runtime metadata.

        Args:
            agent_result: Parameter value.
            requested_mode: Parameter value.
            resolved_mode: Parameter value.
            budget_metadata: Parameter value.

        Returns:
            The resulting value.
        """
        metadata = dict(agent_result.metadata)
        metadata["runtime"] = {
            "requested_mode": requested_mode,
            "resolved_mode": resolved_mode,
            "controls": self._controls.asdict(),
            "soft_budget": dict(budget_metadata),
        }
        return AgentResult(
            output=dict(agent_result.output),
            success=agent_result.success,
            tool_results=list(agent_result.tool_results),
            model_response=agent_result.model_response,
            metadata=metadata,
        )


def _budget_for_result(
    *,
    agent_result: AgentResult,
    controls: RuntimeControls,
    tool_runtime: ToolRuntime,
) -> dict[str, object]:
    """Run budget for result.

    Args:
        agent_result: Parameter value.
        controls: Parameter value.
        tool_runtime: Parameter value.

    Returns:
        The resulting value.
    """
    tracker = _BudgetTracker()
    tracker.add_model_response(agent_result.model_response)
    tracker.add_tool_results(
        tool_results=agent_result.tool_results,
        tool_specs={spec.name: spec for spec in tool_runtime.list_tools()},
    )
    return tracker.as_metadata(controls=controls)
