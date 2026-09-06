"""Internal implementation of evidence-bounded agent paper support."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from design_research_agents._contracts import ExecutionResult
from design_research_agents._paper_observations import (
    coerce_execution_result as _coerce_execution_result,
)
from design_research_agents._paper_observations import has_trace_reference as _has_trace_reference
from design_research_agents._paper_observations import observed_contributions as _observed_contributions
from design_research_agents._paper_observations import redact as _redact
from design_research_agents._paper_observations import resolve_evidence_refs as _resolve_evidence_refs
from design_research_agents._tracing import Tracer
from design_research_agents.agent import DirectLLMCall, MultiStepAgent
from design_research_agents.model_selection import ModelSelector
from design_research_agents.patterns import DebatePattern, PlanExecutePattern, ProposeCriticPattern
from design_research_agents.tools import Toolbox
from design_research_agents.workflow import Workflow

PAPER_CONTRIBUTION_VERSION = "0.1.0"
"""Version of the Experiments-compatible component packet emitted here."""

_PACKAGE = "design-research-agents"
_REACT_REFERENCE: dict[str, Any] = {
    "key": "yao2022react",
    "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
    "year": 2022,
    "url": "https://arxiv.org/abs/2210.03629",
}
_MRKL_REFERENCE: dict[str, Any] = {
    "key": "karpas2022mrkl",
    "title": (
        "MRKL Systems: A modular, neuro-symbolic architecture that combines large language models, "
        "external knowledge sources and discrete reasoning"
    ),
    "year": 2022,
    "url": "https://arxiv.org/abs/2205.00445",
}


def collect_agent_paper_contributions(
    component: object,
    *,
    execution_result: ExecutionResult | Mapping[str, Any] | None = None,
    evidence_refs: Sequence[str] = (),
    component_id: str | None = None,
) -> dict[str, Any]:
    """Describe one agent component and any evidence-backed execution activity.

    Configuration is always reported as ``configured``. Execution, model, tool,
    and workflow claims are reported as ``observed`` only when a durable evidence
    reference is supplied explicitly or recorded as ``metadata.trace_path`` on
    the execution result. The adapter never interprets an agent's generated text.

    Args:
        component: Supported agent, pattern, workflow, toolbox, model selector,
            tracer, or custom component to describe.
        execution_result: Optional completed execution result or persisted-result
            mapping associated with ``component``.
        evidence_refs: Paths or identifiers for durable run evidence.
        component_id: Optional stable identifier overriding the built-in default.

    Returns:
        JSON-compatible contribution packet accepted by
        ``design-research-experiments``.

    Raises:
        TypeError: If ``execution_result`` or ``evidence_refs`` has an invalid type.
        ValueError: If ``component_id`` or an evidence reference is empty.
    """
    description = _describe_component(component)
    resolved_component_id = _resolve_component_id(component_id, description["component_id"])
    source = {
        "package": _PACKAGE,
        "package_version": _package_version(),
        "component_type": description["component_type"],
        "component_id": resolved_component_id,
    }
    references = list(description["references"])
    contributions = [
        {
            "contribution_id": f"agent:{resolved_component_id}:methods",
            "section": "methods",
            "kind": "paragraph",
            "text": description["methods_text"],
            "evidence_basis": "configured",
            "citation_keys": [item["key"] for item in references],
            "evidence_refs": [],
            "metadata": {
                "configuration": _redact(description["configuration"]),
                "reporting_requirements": list(description["reporting_requirements"]),
            },
        }
    ]
    gaps = [
        _rebase_gap(
            gap,
            default_component_id=description["component_id"],
            resolved_component_id=resolved_component_id,
        )
        for gap in description["reporting_gaps"]
    ]

    if execution_result is None:
        gaps.append(
            _gap(
                resolved_component_id,
                "execution-not-provided",
                "No execution result was supplied; reportable runtime activity remains unverified.",
            )
        )
    else:
        result = _coerce_execution_result(execution_result)
        resolved_evidence_refs = _resolve_evidence_refs(evidence_refs, result)
        if resolved_evidence_refs:
            observed_contributions, observed_gaps = _observed_contributions(
                result,
                component_id=resolved_component_id,
                evidence_refs=resolved_evidence_refs,
            )
            contributions.extend(observed_contributions)
            gaps.extend(observed_gaps)
        else:
            gaps.append(
                _gap(
                    resolved_component_id,
                    "execution-evidence-missing",
                    "An execution result was supplied without a durable evidence reference; "
                    "no runtime claim was emitted.",
                )
            )

    if (
        isinstance(component, Tracer)
        and component.enabled
        and not _has_trace_reference(execution_result, evidence_refs)
    ):
        gaps.append(
            _gap(
                resolved_component_id,
                "trace-not-observed",
                "Tracing is configured, but no persisted trace evidence was supplied.",
            )
        )

    return {
        "schema_version": PAPER_CONTRIBUTION_VERSION,
        "source": source,
        "contributions": contributions,
        "references": references,
        "reporting_gaps": _deduplicate_gaps(gaps),
    }


def _package_version() -> str:
    try:
        return version(_PACKAGE)
    except PackageNotFoundError:
        return "0+unknown"


def _resolve_component_id(requested: str | None, default: str) -> str:
    value = default if requested is None else requested
    if not isinstance(value, str) or not value.strip():
        raise ValueError("component_id must be a non-empty string.")
    return value.strip()


def _describe_component(component: object) -> dict[str, Any]:
    if isinstance(component, DirectLLMCall):
        return _describe_direct_llm_call(component)
    if isinstance(component, MultiStepAgent):
        return _describe_multi_step_agent(component)
    if isinstance(component, ProposeCriticPattern):
        return _describe_propose_critic(component)
    if isinstance(component, DebatePattern):
        return _describe_debate(component)
    if isinstance(component, PlanExecutePattern):
        return _describe_plan_execute(component)
    if isinstance(component, Workflow):
        return _describe_workflow(component)
    if isinstance(component, Toolbox):
        return _describe_toolbox(component)
    if isinstance(component, ModelSelector):
        return _describe_model_selector(component)
    if isinstance(component, Tracer):
        return _describe_tracer(component)
    return _describe_custom_component(component)


def _base_description(
    *,
    component_id: str,
    component_type: str,
    methods_text: str,
    configuration: Mapping[str, Any],
    reporting_requirements: Sequence[str],
    references: Sequence[Mapping[str, Any]] = (),
    reporting_gaps: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "component_id": component_id,
        "component_type": component_type,
        "methods_text": methods_text,
        "configuration": dict(configuration),
        "reporting_requirements": tuple(reporting_requirements),
        "references": tuple(dict(item) for item in references),
        "reporting_gaps": [dict(item) for item in reporting_gaps],
    }


def _describe_direct_llm_call(component: DirectLLMCall) -> dict[str, Any]:
    client_snapshot, client_gaps = _client_snapshot(component._llm_client, "agents.direct-llm-call")
    return _base_description(
        component_id="agents.direct-llm-call",
        component_type="agent",
        methods_text=(
            "A one-shot language-model participant was configured as a three-stage workflow that prepares a request, "
            "calls one model, and normalizes the response without a tool runtime."
        ),
        configuration={
            "agent_class": type(component).__name__,
            "client": client_snapshot,
            "system_prompt_configured": component._default_system_prompt is not None,
            "temperature": component._temperature,
            "max_tokens": component._max_tokens,
            "provider_options": component._provider_options,
            "tracing_configured": component._tracer is not None,
        },
        reporting_requirements=(
            "Report the provider and model identifier used for each run.",
            "Report sampling controls and any provider-specific options that affect generation.",
            "Link execution claims to persisted run evidence.",
        ),
        reporting_gaps=client_gaps,
    )


def _describe_multi_step_agent(component: MultiStepAgent) -> dict[str, Any]:
    strategy = component._strategy
    component_key = "agents.multi-step"
    client_snapshot, client_gaps = _client_snapshot(getattr(strategy, "_llm_client", None), component_key)
    runtime = getattr(strategy, "_tool_runtime", None)
    tools = _tool_specs(runtime)
    configuration: dict[str, Any] = {
        "agent_class": type(component).__name__,
        "mode": component._mode,
        "strategy_class": type(strategy).__name__,
        "client": client_snapshot,
        "max_steps": getattr(strategy, "_max_steps", None),
        "stop_on_step_failure": getattr(strategy, "_stop_on_step_failure", None),
        "allowed_tools": getattr(strategy, "_allowed_tools", None),
        "available_tools": tools,
        "max_tool_calls_per_step": getattr(strategy, "_max_tool_calls_per_step", None),
        "execution_timeout_seconds": getattr(strategy, "_execution_timeout_seconds", None),
        "tracing_configured": getattr(strategy, "_tracer", None) is not None,
    }
    references = (_REACT_REFERENCE, _MRKL_REFERENCE) if runtime is not None else (_REACT_REFERENCE,)
    return _base_description(
        component_id=component_key,
        component_type="agent",
        methods_text=(
            f"A multi-step language-model participant was configured in {component._mode!r} mode with a bounded "
            "controller loop and explicit termination behavior. Available tools are configuration only until a run "
            "records an invocation."
        ),
        configuration=configuration,
        reporting_requirements=(
            "Report the execution mode, step limit, stop condition, and model configuration.",
            "Distinguish tools made available to the participant from tools actually invoked.",
            "Report tool failures and termination reasons alongside successful calls.",
        ),
        references=references,
        reporting_gaps=client_gaps,
    )


def _describe_propose_critic(component: ProposeCriticPattern) -> dict[str, Any]:
    return _describe_iterative_pattern(
        component,
        component_id="agents.pattern.propose-critic",
        pattern_name="propose-and-critique",
        iteration_field="_max_iterations",
        iteration_label="max_iterations",
        role_fields=("_proposer_delegate", "_critic_delegate"),
    )


def _describe_debate(component: DebatePattern) -> dict[str, Any]:
    return _describe_iterative_pattern(
        component,
        component_id="agents.pattern.debate",
        pattern_name="role-based debate",
        iteration_field="_max_rounds",
        iteration_label="max_rounds",
        role_fields=("_affirmative_delegate", "_negative_delegate", "_judge_delegate"),
    )


def _describe_plan_execute(component: PlanExecutePattern) -> dict[str, Any]:
    description = _describe_iterative_pattern(
        component,
        component_id="agents.pattern.plan-execute",
        pattern_name="plan-and-execute",
        iteration_field="_max_iterations",
        iteration_label="max_iterations",
        role_fields=("_planner_delegate", "_executor_delegate"),
    )
    description["configuration"]["max_tool_calls_per_step"] = component._max_tool_calls_per_step
    description["references"] = (_REACT_REFERENCE, _MRKL_REFERENCE)
    return description


def _describe_iterative_pattern(
    component: object,
    *,
    component_id: str,
    pattern_name: str,
    iteration_field: str,
    iteration_label: str,
    role_fields: Sequence[str],
) -> dict[str, Any]:
    client_snapshot, client_gaps = _client_snapshot(getattr(component, "_llm_client", None), component_id)
    runtime = getattr(component, "_tool_runtime", None)
    return _base_description(
        component_id=component_id,
        component_type="agent-pattern",
        methods_text=(
            f"A {pattern_name} orchestration was configured with bounded iteration and explicit participant roles. "
            "Any attached tool runtime describes availability only until invocation evidence is recorded."
        ),
        configuration={
            "pattern_class": type(component).__name__,
            iteration_label: getattr(component, iteration_field),
            "client": client_snapshot,
            "available_tools": _tool_specs(runtime),
            "delegate_overrides": {
                field.removeprefix("_").removesuffix("_delegate"): getattr(component, field) is not None
                for field in role_fields
            },
            "request_id_prefix": getattr(component, "_default_request_id_prefix", None),
            "tracing_configured": getattr(component, "_tracer", None) is not None,
        },
        reporting_requirements=(
            f"Report the {iteration_label.replace('_', ' ')}, role definitions, and stopping rule.",
            "Report the model configuration used by each role or delegate.",
            "Distinguish configured tool availability from observed invocation.",
        ),
        references=(_REACT_REFERENCE,),
        reporting_gaps=client_gaps,
    )


def _describe_workflow(component: Workflow) -> dict[str, Any]:
    steps = [_workflow_step_snapshot(step) for step in component._steps]
    return _base_description(
        component_id="agents.workflow",
        component_type="workflow",
        methods_text=(
            "A declarative workflow graph was configured with typed steps, explicit dependencies, execution mode, "
            "and dependency-failure policy. The graph description does not imply that any step executed."
        ),
        configuration={
            "workflow_class": type(component).__name__,
            "steps": steps,
            "input_schema": component._input_schema,
            "output_schema": component._output_schema,
            "prompt_context_key": component._prompt_context_key,
            "execution_mode": component._default_execution_mode,
            "failure_policy": component._default_failure_policy,
            "request_id_prefix": component._default_request_id_prefix,
        },
        reporting_requirements=(
            "Report step identifiers, step types, dependencies, execution mode, and failure policy.",
            "Report observed step status and order from persisted execution evidence.",
            "Identify configured tool steps separately from successful or failed invocations.",
        ),
    )


def _describe_toolbox(component: Toolbox) -> dict[str, Any]:
    tools = _tool_specs(component)
    source_counts = Counter(str(item.get("source", "unknown")) for item in tools)
    return _base_description(
        component_id="agents.toolbox",
        component_type="tool-runtime",
        methods_text=(
            f"A tool runtime exposed {len(tools)} configured tool(s) across explicit runtime sources. Exposure made "
            "a tool available to agents but does not establish that it was invoked."
        ),
        configuration={
            "runtime_class": type(component).__name__,
            "available_tools": tools,
            "source_counts": dict(sorted(source_counts.items())),
        },
        reporting_requirements=(
            "Report available tool names, sources, permissions, and declared side effects.",
            "Report actual invocation status only from persisted ToolResult evidence.",
            "Include failed calls and warnings in run accounting.",
        ),
        references=(_MRKL_REFERENCE,),
    )


def _describe_model_selector(component: ModelSelector) -> dict[str, Any]:
    policy = component._policy
    config = asdict(policy.config)
    return _base_description(
        component_id="agents.model-selector",
        component_type="model-selector",
        methods_text=(
            "A model-selection policy was configured to filter and rank local or remote candidates under explicit "
            "resource, cost, latency, and provider preferences."
        ),
        configuration={
            "selector_class": type(component).__name__,
            "policy": config,
            "catalog_signature": policy.catalog.signature(),
            "candidate_count": len(policy.catalog.models),
            "custom_local_resolver": component._local_client_resolver is not None,
        },
        reporting_requirements=(
            "Report selection intent, constraints, hardware snapshot, policy identifier, and catalog signature.",
            "Report the selected provider and model from the recorded selection decision.",
        ),
    )


def _describe_tracer(component: Tracer) -> dict[str, Any]:
    return _base_description(
        component_id="agents.tracer",
        component_type="tracer",
        methods_text=(
            "Execution tracing was explicitly configured with independently controlled JSONL and console sinks."
        ),
        configuration={
            "tracer_class": type(component).__name__,
            "enabled": component.enabled,
            "trace_dir": str(component.trace_dir),
            "jsonl_enabled": component.enable_jsonl,
            "console_enabled": component.enable_console,
        },
        reporting_requirements=(
            "Report whether tracing was enabled and which durable sink was configured.",
            "Link trace claims to a persisted trace path or other durable run evidence.",
        ),
    )


def _describe_custom_component(component: object) -> dict[str, Any]:
    component_name = type(component).__name__
    component_id = f"agents.custom.{_slug(component_name)}"
    return _base_description(
        component_id=component_id,
        component_type="custom-agent-component",
        methods_text=(
            f"A custom agent component of type {component_name!r} was configured. No package-owned adapter is "
            "registered for its internal configuration."
        ),
        configuration={"component_class": component_name},
        reporting_requirements=(
            "Provide a stable component identifier and a factual component-specific Methods description.",
            "Record model, tool, workflow, and trace configuration through an explicit adapter.",
        ),
        reporting_gaps=(
            _gap(
                component_id,
                "custom-component-metadata-incomplete",
                "No explicit package-owned adapter is registered for this custom component.",
            ),
        ),
    )


def _client_snapshot(client: object | None, component_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if client is None:
        return (
            {"client_type": "unknown"},
            [_gap(component_id, "model-configuration-missing", "No model client configuration was available.")],
        )
    snapshot: dict[str, Any] = {"client_type": type(client).__name__}
    describe = getattr(client, "describe", None)
    if callable(describe):
        try:
            described = describe()
        except Exception:
            described = None
        if isinstance(described, Mapping):
            snapshot["description"] = described
    default_model = getattr(client, "default_model", None)
    if callable(default_model):
        try:
            model = default_model()
        except Exception:
            model = None
        if isinstance(model, str) and model.strip():
            snapshot["default_model"] = model
    if len(snapshot) > 1:
        return _redact(snapshot), []
    return (
        snapshot,
        [
            _gap(
                component_id,
                "model-configuration-incomplete",
                "The configured model client did not expose a stable description or default model identifier.",
            )
        ],
    )


def _tool_specs(runtime: object | None) -> list[dict[str, Any]]:
    if runtime is None:
        return []
    list_tools = getattr(runtime, "list_tools", None)
    if not callable(list_tools):
        return []
    try:
        specs = list_tools()
    except Exception:
        return []
    normalized: list[dict[str, Any]] = []
    for spec in specs:
        metadata = getattr(spec, "metadata", None)
        side_effects = getattr(metadata, "side_effects", None)
        normalized.append(
            {
                "name": str(getattr(spec, "name", type(spec).__name__)),
                "source": str(getattr(metadata, "source", "unknown")),
                "permissions": sorted(str(item) for item in getattr(spec, "permissions", ())),
                "risky": bool(getattr(metadata, "risky", False)),
                "side_effects": {
                    "filesystem_read": bool(getattr(side_effects, "filesystem_read", False)),
                    "filesystem_write": bool(getattr(side_effects, "filesystem_write", False)),
                    "network": bool(getattr(side_effects, "network", False)),
                    "commands": sorted(str(item) for item in getattr(side_effects, "commands", ())),
                },
            }
        )
    return sorted(normalized, key=lambda item: (item["name"], item["source"]))


def _workflow_step_snapshot(step: object) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "step_id": str(getattr(step, "step_id", "unknown")),
        "step_type": type(step).__name__,
        "dependencies": sorted(str(item) for item in getattr(step, "dependencies", ())),
    }
    tool_name = getattr(step, "tool_name", None)
    if isinstance(tool_name, str) and tool_name:
        snapshot["configured_tool_name"] = tool_name
    nested_steps = getattr(step, "steps", None)
    if isinstance(nested_steps, Sequence) and not isinstance(nested_steps, (str, bytes)):
        snapshot["nested_steps"] = [_workflow_step_snapshot(item) for item in nested_steps]
    return snapshot


def _gap(
    component_id: str,
    suffix: str,
    message: str,
    *,
    evidence_refs: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "gap_id": f"agent:{component_id}:{suffix}",
        "section": "methods",
        "message": message,
        "evidence_refs": list(evidence_refs),
    }


def _deduplicate_gaps(gaps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for gap in gaps:
        gap_id = str(gap.get("gap_id", ""))
        selected.setdefault(gap_id, dict(gap))
    return list(selected.values())


def _rebase_gap(
    gap: Mapping[str, Any],
    *,
    default_component_id: str,
    resolved_component_id: str,
) -> dict[str, Any]:
    normalized = dict(gap)
    default_prefix = f"agent:{default_component_id}:"
    gap_id = str(normalized.get("gap_id", ""))
    if gap_id.startswith(default_prefix):
        normalized["gap_id"] = f"agent:{resolved_component_id}:{gap_id.removeprefix(default_prefix)}"
    return normalized


def _slug(value: str) -> str:
    normalized = re.sub(r"(?<!^)(?=[A-Z])", "-", value).lower()
    normalized = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")
    return normalized or "component"


__all__ = ["PAPER_CONTRIBUTION_VERSION", "collect_agent_paper_contributions"]
