"""Static Mermaid rendering helpers for workflow topology inspection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._workflow import LogicStep, LoopStep, WorkflowStep
from design_research_agents._runtime._workflow import prepare_workflow_graph, validate_no_cycles

from ._diagram_common import (
    _escape_label,
    _normalize_direction,
    _qualified_step_id,
    _route_label,
    _step_label,
)
from ._diagram_svg import render_workflow_as_svg

__all__ = ["render_workflow_as_mermaid", "render_workflow_as_svg"]


class _MermaidBuilder:
    """Collect Mermaid lines with stable synthetic node identifiers."""

    def __init__(self, *, direction: str) -> None:
        self.direction = direction
        self.lines: list[str] = [f"flowchart {direction}"]
        self._indent = 1
        self._step_counter = 0
        self._loop_entry_counter = 0
        self._subgraph_counter = 0

    def line(self, value: str) -> None:
        """Append one indented Mermaid line."""
        self.lines.append(f"{'    ' * self._indent}{value}")

    def next_step_node_id(self) -> str:
        """Return the next synthetic step node id."""
        self._step_counter += 1
        return f"step_{self._step_counter}"

    def next_loop_entry_id(self) -> str:
        """Return the next synthetic loop entry id."""
        self._loop_entry_counter += 1
        return f"loop_entry_{self._loop_entry_counter}"

    def next_subgraph_id(self) -> str:
        """Return the next synthetic subgraph id."""
        self._subgraph_counter += 1
        return f"loop_body_{self._subgraph_counter}"

    def begin_subgraph(self, *, subgraph_id: str, label: str) -> None:
        """Start a Mermaid subgraph block."""
        self.line(f'subgraph {subgraph_id}["{_escape_label(label)}"]')
        self._indent += 1
        self.line("direction TD")

    def end_subgraph(self) -> None:
        """Close the current Mermaid subgraph block."""
        self._indent = max(1, self._indent - 1)
        self.line("end")

    def render(self) -> str:
        """Return the final Mermaid document."""
        return "\n".join(self.lines)


def render_workflow_as_mermaid(steps: Sequence[WorkflowStep], *, direction: str = "TD") -> str:
    """Render one static workflow step graph as Mermaid flowchart text."""
    normalized_direction = _normalize_direction(direction)
    prepared = prepare_workflow_graph(steps)
    validate_no_cycles(prepared.step_map, prepared.dependencies)

    builder = _MermaidBuilder(direction=normalized_direction)
    builder.line('workflow_entry["Workflow Entrypoint"]')
    _render_sequence(
        builder=builder,
        steps=steps,
        entry_node_id="workflow_entry",
        qualified_prefix=(),
    )
    return builder.render()


def _render_sequence(
    *,
    builder: _MermaidBuilder,
    steps: Sequence[WorkflowStep],
    entry_node_id: str,
    qualified_prefix: tuple[str, ...],
) -> tuple[str, ...]:
    """Render one workflow step sequence and return terminal node ids."""
    if not steps:
        return ()

    prepared = prepare_workflow_graph(steps)
    validate_no_cycles(prepared.step_map, prepared.dependencies)
    step_nodes = _declare_step_nodes(
        builder=builder,
        steps=steps,
        qualified_prefix=qualified_prefix,
    )
    loop_entry_nodes = _render_loop_subgraphs(
        builder=builder,
        steps=steps,
        qualified_prefix=qualified_prefix,
    )
    _render_sequence_edges(
        builder=builder,
        steps=steps,
        prepared_dependencies=prepared.dependencies,
        step_nodes=step_nodes,
        loop_entry_nodes=loop_entry_nodes,
        entry_node_id=entry_node_id,
    )
    return _terminal_node_ids(
        steps=steps,
        step_nodes=step_nodes,
        dependents=prepared.dependents,
    )


def _declare_step_nodes(
    *,
    builder: _MermaidBuilder,
    steps: Sequence[WorkflowStep],
    qualified_prefix: tuple[str, ...],
) -> dict[str, str]:
    """Declare Mermaid nodes for one ordered step sequence."""
    step_nodes: dict[str, str] = {}
    for step in steps:
        step_node_id = builder.next_step_node_id()
        step_nodes[step.step_id] = step_node_id
        qualified_id = _qualified_step_id(qualified_prefix, step.step_id)
        builder.line(f'{step_node_id}["{_escape_label(_step_label(step, qualified_id))}"]')
    return step_nodes


def _render_loop_subgraphs(
    *,
    builder: _MermaidBuilder,
    steps: Sequence[WorkflowStep],
    qualified_prefix: tuple[str, ...],
) -> dict[str, str]:
    """Render nested Mermaid subgraphs for loop bodies."""
    loop_entry_nodes: dict[str, str] = {}
    for step in steps:
        if not isinstance(step, LoopStep):
            continue
        loop_entry_id = builder.next_loop_entry_id()
        loop_entry_nodes[step.step_id] = loop_entry_id
        qualified_id = _qualified_step_id(qualified_prefix, step.step_id)
        builder.begin_subgraph(
            subgraph_id=builder.next_subgraph_id(),
            label=f"Loop Body: {qualified_id}",
        )
        builder.line(f'{loop_entry_id}["{_escape_label(f"{qualified_id} iteration entry")}"]')
        terminal_nodes = _render_sequence(
            builder=builder,
            steps=step.steps,
            entry_node_id=loop_entry_id,
            qualified_prefix=(*qualified_prefix, step.step_id),
        )
        for terminal_node_id in terminal_nodes:
            builder.line(f'{terminal_node_id} -. "next iteration" .-> {loop_entry_id}')
        builder.end_subgraph()
    return loop_entry_nodes


def _render_sequence_edges(
    *,
    builder: _MermaidBuilder,
    steps: Sequence[WorkflowStep],
    prepared_dependencies: Mapping[str, tuple[str, ...]],
    step_nodes: Mapping[str, str],
    loop_entry_nodes: Mapping[str, str],
    entry_node_id: str,
) -> None:
    """Render dependency, loop, and route edges for one step sequence."""
    for step in steps:
        step_node_id = step_nodes[step.step_id]
        _render_dependency_edges(
            builder=builder,
            dependencies=prepared_dependencies.get(step.step_id, ()),
            step_nodes=step_nodes,
            step_node_id=step_node_id,
            entry_node_id=entry_node_id,
        )
        _render_loop_edge(
            builder=builder,
            step=step,
            step_node_id=step_node_id,
            loop_entry_nodes=loop_entry_nodes,
        )
        _render_route_edges(
            builder=builder,
            step=step,
            step_node_id=step_node_id,
            step_nodes=step_nodes,
        )


def _render_dependency_edges(
    *,
    builder: _MermaidBuilder,
    dependencies: Sequence[str],
    step_nodes: Mapping[str, str],
    step_node_id: str,
    entry_node_id: str,
) -> None:
    """Render dependency edges into one step node."""
    if not dependencies:
        builder.line(f"{entry_node_id} --> {step_node_id}")
        return
    for dependency_step_id in dependencies:
        dependency_node_id = step_nodes[dependency_step_id]
        builder.line(f"{dependency_node_id} --> {step_node_id}")


def _render_loop_edge(
    *,
    builder: _MermaidBuilder,
    step: WorkflowStep,
    step_node_id: str,
    loop_entry_nodes: Mapping[str, str],
) -> None:
    """Render one outer-loop edge when the step is a loop."""
    if not isinstance(step, LoopStep):
        return
    maybe_loop_entry_id = loop_entry_nodes.get(step.step_id)
    if maybe_loop_entry_id is None:
        return
    builder.line(f'{step_node_id} -. "iterate" .-> {maybe_loop_entry_id}')


def _render_route_edges(
    *,
    builder: _MermaidBuilder,
    step: WorkflowStep,
    step_node_id: str,
    step_nodes: Mapping[str, str],
) -> None:
    """Render one step's conditional route edges when present."""
    if not isinstance(step, LogicStep) or not isinstance(step.route_map, Mapping):
        return
    for route_key, route_targets in step.route_map.items():
        route_label = _route_label(route_key)
        for route_target in route_targets:
            target_node_id = step_nodes.get(route_target)
            if target_node_id is None:
                continue
            builder.line(f'{step_node_id} -. "{_escape_label(route_label)}" .-> {target_node_id}')


def _terminal_node_ids(
    *,
    steps: Sequence[WorkflowStep],
    step_nodes: Mapping[str, str],
    dependents: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Return node ids for terminal steps in one prepared sequence."""
    return tuple(step_nodes[step.step_id] for step in steps if not dependents.get(step.step_id))
