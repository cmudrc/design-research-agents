"""Reusable ``propose_critic`` orchestration chunk."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from uuid import uuid4

from design_research_agents.agent import AgentRuntime
from design_research_agents.agent.runtime_controls import RuntimeControls
from design_research_agents.contracts.agent import Agent, AgentResult, AgentStreamEvent
from design_research_agents.contracts.llm import LLMClient
from design_research_agents.contracts.tools import ToolRuntime
from design_research_agents.tracing import Tracer


class ProposeAndCritiqueWorkflow(Agent):
    """Configured workflow chunk for ``propose_critic`` mode."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        tool_runtime: ToolRuntime,
        controls: RuntimeControls | None = None,
        propose_critic_proposer_system_prompt: str | None = None,
        propose_critic_proposer_user_prompt_template: str | None = None,
        propose_critic_critic_system_prompt: str | None = None,
        propose_critic_critic_user_prompt_template: str | None = None,
        default_request_id_prefix: str | None = None,
        default_dependencies: Mapping[str, object] | None = None,
        tracer: Tracer | None = None,
    ) -> None:
        """Store dependencies and initialize the underlying runtime.

        Args:
            llm_client: LLM client to use for this workflow.
            tool_runtime: Tool runtime to use for this workflow.
            controls: Optional default runtime controls for all runs of this workflow.
            propose_critic_proposer_system_prompt: Optional system prompt to use for the proposer.
            propose_critic_proposer_user_prompt_template: Optional user prompt template to use
                for the proposer.
            propose_critic_critic_system_prompt: Optional system prompt to use for the critic.
            propose_critic_critic_user_prompt_template: Optional user prompt template to use
                for the critic.
            default_request_id_prefix: Optional prefix to use when generating request IDs
                for runs of this workflow that don't provide their own request ID. Must
                be non-empty when provided.
            default_dependencies: Optional mapping of default dependencies to provide for
                all runs of this workflow, which can be overridden by dependencies provided
                at run time.
            tracer: Optional tracer for emitting events during execution.
        """
        self._default_request_id_prefix = _normalize_request_id_prefix(default_request_id_prefix)
        self._default_dependencies = dict(default_dependencies or {})
        self._runtime = AgentRuntime(
            llm_client=llm_client,
            tool_runtime=tool_runtime,
            mode="propose_critic",
            controls=controls,
            propose_critic_proposer_system_prompt=propose_critic_proposer_system_prompt,
            propose_critic_proposer_user_prompt_template=propose_critic_proposer_user_prompt_template,
            propose_critic_critic_system_prompt=propose_critic_critic_system_prompt,
            propose_critic_critic_user_prompt_template=propose_critic_critic_user_prompt_template,
            tracer=tracer,
        )

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> AgentResult:
        """Execute one propose-and-critique orchestration run.

        Args:
            prompt: The user prompt to run.
            request_id: Optional request ID to use for this run. If not provided, a request
                ID will be generated using the default prefix configured for this workflow (if
                any) and a random suffix.
            dependencies: Optional mapping of dependencies to provide for this run, which will
                be merged with the default dependencies configured for this workflow (if any),
                with run dependencies taking precedence over default dependencies in case of
                overlap.
        """
        resolved_request_id = _resolve_request_id(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        return self._runtime.run(
            prompt,
            request_id=resolved_request_id,
            dependencies=_merge_dependencies(
                default_dependencies=self._default_dependencies,
                run_dependencies=dependencies,
            ),
        )

    def run_stream(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> Iterator[AgentStreamEvent]:
        """Execute one run and emit streaming events.

        Args:
            prompt: The user prompt to run.
            request_id: Optional request ID to use for this run. If not provided, a request
                ID will be generated using the default prefix configured for this workflow (if
                any) and a random suffix.
            dependencies: Optional mapping of dependencies to provide for this run, which will
                be merged with the default dependencies configured for this workflow (if any),
                with run dependencies taking precedence over default dependencies in case of
                overlap.
        """
        resolved_request_id = _resolve_request_id(
            request_id=request_id,
            default_prefix=self._default_request_id_prefix,
        )
        yield from self._runtime.run_stream(
            prompt,
            request_id=resolved_request_id,
            dependencies=_merge_dependencies(
                default_dependencies=self._default_dependencies,
                run_dependencies=dependencies,
            ),
        )


def _merge_dependencies(
    *,
    default_dependencies: Mapping[str, object],
    run_dependencies: Mapping[str, object] | None,
) -> dict[str, object]:
    merged = dict(default_dependencies)
    if run_dependencies is not None:
        merged.update(run_dependencies)
    return merged


def _normalize_request_id_prefix(default_request_id_prefix: str | None) -> str | None:
    if default_request_id_prefix is None:
        return None
    normalized = default_request_id_prefix.strip()
    if not normalized:
        raise ValueError("default_request_id_prefix must be non-empty when provided.")
    return normalized


def _resolve_request_id(*, request_id: str | None, default_prefix: str | None) -> str | None:
    if request_id is not None and request_id.strip():
        return request_id
    if default_prefix is None:
        return request_id
    return f"{default_prefix}:{uuid4().hex}"


__all__ = [
    "ProposeAndCritiqueWorkflow",
]
