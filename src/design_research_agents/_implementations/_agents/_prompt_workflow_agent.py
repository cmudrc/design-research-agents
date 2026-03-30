"""Prompt-driven workflow agent for packaged-problem experiment studies."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from design_research_agents._contracts._delegate import Delegate
from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents.workflow import CompiledExecution, Workflow

_DEFAULT_REQUEST_ID = "prompt-workflow-agent"


class PromptWorkflowAgent(Delegate):
    """Wrap one prompt-mode workflow as a first-class executable agent."""

    def __init__(
        self,
        *,
        workflow: Workflow,
        prompt_builder: Callable[[object, object, object], str],
    ) -> None:
        """Store the wrapped workflow and study-aware prompt builder."""
        self.workflow = workflow
        self._prompt_builder = prompt_builder

    def compile(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> CompiledExecution:
        """Compile one study-oriented workflow execution."""
        resolved_prompt = self._resolve_prompt(prompt=prompt, dependencies=dependencies)
        resolved_request_id = request_id or _DEFAULT_REQUEST_ID
        resolved_dependencies = dict(dependencies or {})
        self.workflow = self.workflow
        return CompiledExecution(
            workflow=self.workflow,
            input=resolved_prompt,
            request_id=resolved_request_id,
            workflow_request_id=resolved_request_id,
            dependencies=resolved_dependencies,
            delegate_name="PromptWorkflowAgent",
            trace_input={"prompt": resolved_prompt},
        )

    def run(
        self,
        prompt: str | object,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Execute one compiled study-oriented workflow run."""
        return self.compile(
            prompt=prompt,
            request_id=request_id,
            dependencies=dependencies,
        ).run()

    def _resolve_prompt(
        self,
        *,
        prompt: str | object,
        dependencies: Mapping[str, object] | None,
    ) -> str:
        """Resolve one workflow prompt from study dependencies or a direct fallback."""
        if dependencies is not None:
            problem_packet = dependencies.get("problem_packet")
            run_spec = dependencies.get("run_spec")
            condition = dependencies.get("condition")
            if problem_packet is not None and run_spec is not None and condition is not None:
                built_prompt = self._prompt_builder(problem_packet, run_spec, condition)
                normalized_prompt = built_prompt.strip()
                if not normalized_prompt:
                    raise ValueError("PromptWorkflowAgent prompt_builder returned an empty prompt.")
                return normalized_prompt

        if isinstance(prompt, str):
            normalized_prompt = prompt.strip()
            if normalized_prompt:
                return normalized_prompt

        raise ValueError(
            "PromptWorkflowAgent requires study dependencies "
            "(`problem_packet`, `run_spec`, `condition`) or a non-empty prompt."
        )


__all__ = ["PromptWorkflowAgent"]
