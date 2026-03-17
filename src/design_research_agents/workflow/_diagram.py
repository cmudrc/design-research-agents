"""Static Mermaid rendering helpers for workflow topology inspection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents._contracts._workflow import LogicStep, LoopStep, WorkflowStep
from design_research_agents._runtime._workflow import prepare_workflow_graph, validate_no_cycles

from ._diagram_common import (
    _delegate_nested_steps,
    _escape_label,
    _normalize_direction,
    _qualified_step_id,
    _route_label,
    _step_label,
)
from ._diagram_svg import render_workflow_as_svg

__all__ = ["render_workflow_as_mermaid", "render_workflow_as_svg"]


@dataclass(slots=True, frozen=True)
class _NestedMermaidSpec:
    """Nested Mermaid subgraph description for loop or delegate expansion."""

    subgraph_label: str
    entry_label: str
    edge_label: str
    terminal_edge_label: str | None
    terminal_edge_style: str
    steps: tuple[WorkflowStep, ...]
    qualified_prefix: tuple[str, ...]
    entry_id_kind: str


class _MermaidBuilder:
    """Collect Mermaid lines with stable synthetic node identifiers."""

    def __init__(self, *, direction: str) -> None:
        self.direction = direction
        self.lines: list[str] = [f"flowchart {direction}"]
        self._indent = 1
        self._step_counter = 0
        self._loop_entry_counter = 0
        self._delegate_entry_counter = 0
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

    def next_delegate_entry_id(self) -> str:
        """Return the next synthetic delegate entry id."""
        self._delegate_entry_counter += 1
        return f"delegate_entry_{self._delegate_entry_counter}"

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
    nested_entry_nodes = _render_nested_subgraphs(
        builder=builder,
        steps=steps,
        qualified_prefix=qualified_prefix,
    )
    _render_sequence_edges(
        builder=builder,
        steps=steps,
        prepared_dependencies=prepared.dependencies,
        step_nodes=step_nodes,
        nested_entry_nodes=nested_entry_nodes,
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


def _render_nested_subgraphs(
    *,
    builder: _MermaidBuilder,
    steps: Sequence[WorkflowStep],
    qualified_prefix: tuple[str, ...],
) -> dict[str, str]:
    """Render nested Mermaid subgraphs for loop and delegate bodies."""
    nested_entry_nodes: dict[str, str] = {}
    for step in steps:
        qualified_id = _qualified_step_id(qualified_prefix, step.step_id)
        nested_spec = _nested_mermaid_spec(
            step=step,
            qualified_step_id=qualified_id,
            qualified_prefix=qualified_prefix,
        )
        if nested_spec is None:
            continue
        nested_entry_id = (
            builder.next_loop_entry_id() if nested_spec.entry_id_kind == "loop" else builder.next_delegate_entry_id()
        )
        nested_entry_nodes[step.step_id] = nested_entry_id
        builder.begin_subgraph(
            subgraph_id=builder.next_subgraph_id(),
            label=nested_spec.subgraph_label,
        )
        builder.line(f'{nested_entry_id}["{_escape_label(nested_spec.entry_label)}"]')
        terminal_nodes = _render_sequence(
            builder=builder,
            steps=nested_spec.steps,
            entry_node_id=nested_entry_id,
            qualified_prefix=nested_spec.qualified_prefix,
        )
        if nested_spec.terminal_edge_label is not None:
            for terminal_node_id in terminal_nodes:
                if nested_spec.terminal_edge_style == "dashed":
                    builder.line(
                        f'{terminal_node_id} -. "{_escape_label(nested_spec.terminal_edge_label)}" .-> '
                        f"{nested_entry_id}"
                    )
                else:
                    builder.line(
                        f'{terminal_node_id} -- "{_escape_label(nested_spec.terminal_edge_label)}" --> '
                        f"{nested_entry_id}"
                    )
        builder.end_subgraph()
    return nested_entry_nodes


def _render_sequence_edges(
    *,
    builder: _MermaidBuilder,
    steps: Sequence[WorkflowStep],
    prepared_dependencies: Mapping[str, tuple[str, ...]],
    step_nodes: Mapping[str, str],
    nested_entry_nodes: Mapping[str, str],
    entry_node_id: str,
) -> None:
    """Render dependency, loop, and route edges for one step sequence."""
    step_lookup = {step.step_id: step for step in steps}
    for step in steps:
        step_node_id = step_nodes[step.step_id]
        _render_dependency_edges(
            builder=builder,
            dependencies=prepared_dependencies.get(step.step_id, ()),
            target_step_id=step.step_id,
            step_lookup=step_lookup,
            step_nodes=step_nodes,
            step_node_id=step_node_id,
            entry_node_id=entry_node_id,
        )
        _render_loop_edge(
            builder=builder,
            step=step,
            step_node_id=step_node_id,
            nested_entry_nodes=nested_entry_nodes,
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
    target_step_id: str,
    step_lookup: Mapping[str, WorkflowStep],
    step_nodes: Mapping[str, str],
    step_node_id: str,
    entry_node_id: str,
) -> None:
    """Render dependency edges into one step node."""
    if not dependencies:
        builder.line(f"{entry_node_id} --> {step_node_id}")
        return
    for dependency_step_id in dependencies:
        dependency_step = step_lookup.get(dependency_step_id)
        if dependency_step is not None and _is_routed_dependency_edge(
            dependency_step=dependency_step,
            target_step_id=target_step_id,
        ):
            continue
        dependency_node_id = step_nodes[dependency_step_id]
        builder.line(f"{dependency_node_id} --> {step_node_id}")


def _render_loop_edge(
    *,
    builder: _MermaidBuilder,
    step: WorkflowStep,
    step_node_id: str,
    nested_entry_nodes: Mapping[str, str],
) -> None:
    """Render one edge from a step to its nested expansion when present."""
    nested_entry_id = nested_entry_nodes.get(step.step_id)
    if nested_entry_id is None:
        return
    edge_label = "iterate" if isinstance(step, LoopStep) else "delegate"
    builder.line(f'{step_node_id} -. "{edge_label}" .-> {nested_entry_id}')


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


def _nested_mermaid_spec(
    *,
    step: WorkflowStep,
    qualified_step_id: str,
    qualified_prefix: tuple[str, ...],
) -> _NestedMermaidSpec | None:
    """Return Mermaid nested-layout metadata for loops or delegates."""
    if isinstance(step, LoopStep):
        return _NestedMermaidSpec(
            subgraph_label=f"Loop Body: {qualified_step_id}",
            entry_label=f"{qualified_step_id} iteration entry",
            edge_label="iterate",
            terminal_edge_label="next iteration",
            terminal_edge_style="dashed",
            steps=tuple(step.steps),
            qualified_prefix=(*qualified_prefix, step.step_id),
            entry_id_kind="loop",
        )

    nested_steps = _delegate_nested_steps(step)
    if nested_steps is None:
        return None
    return _NestedMermaidSpec(
        subgraph_label=f"Delegate Workflow: {qualified_step_id}",
        entry_label=f"{qualified_step_id} delegate entry",
        edge_label="delegate",
        terminal_edge_label=None,
        terminal_edge_style="solid",
        steps=nested_steps,
        qualified_prefix=(*qualified_prefix, step.step_id),
        entry_id_kind="delegate",
    )


def _is_routed_dependency_edge(*, dependency_step: WorkflowStep, target_step_id: str) -> bool:
    """Return True when one dependency is already represented by a route edge."""
    if not isinstance(dependency_step, LogicStep) or not isinstance(dependency_step.route_map, Mapping):
        return False
    return any(target_step_id in route_targets for route_targets in dependency_step.route_map.values())


def _terminal_node_ids(
    *,
    steps: Sequence[WorkflowStep],
    step_nodes: Mapping[str, str],
    dependents: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Return node ids for terminal steps in one prepared sequence."""
    return tuple(step_nodes[step.step_id] for step in steps if not dependents.get(step.step_id))
