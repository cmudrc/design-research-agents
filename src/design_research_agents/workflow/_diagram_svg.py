"""Static SVG rendering helpers for workflow topology inspection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from design_research_agents._contracts._workflow import (
    DelegateBatchStep,
    DelegateStep,
    LogicStep,
    LoopStep,
    MemoryReadStep,
    MemoryWriteStep,
    ModelStep,
    ToolStep,
    WorkflowStep,
)
from design_research_agents._runtime._workflow import prepare_workflow_graph, validate_no_cycles

from ._diagram_common import (
    _delegate_nested_steps,
    _normalize_direction,
    _qualified_step_id,
    _route_label,
    _step_label,
)
from ._diagram_svg_render import _render_svg_document
from ._diagram_svg_types import (
    _SVG_LINE_HEIGHT,
    _SVG_NODE_VERTICAL_PADDING,
    _SvgEdge,
    _SvgGroup,
    _SvgLayout,
    _SvgNestedLayout,
    _SvgNode,
    _SvgRowLayout,
)

_SVG_NODE_HORIZONTAL_PADDING = 18.0
_SVG_CHARACTER_WIDTH = 7.0
_SVG_VERTICAL_GAP = 40.0
_SVG_HORIZONTAL_GAP = 84.0
_SVG_GROUP_PADDING = 18.0
_SVG_GROUP_LABEL_HEIGHT = 28.0


class _SvgLayoutBuilder:
    """Allocate stable SVG node/group identifiers."""

    def __init__(self) -> None:
        self._step_counter = 0
        self._loop_entry_counter = 0
        self._delegate_entry_counter = 0

    def next_step_node_id(self) -> str:
        """Return the next synthetic step node id."""
        self._step_counter += 1
        return f"step_{self._step_counter}"

    def next_loop_entry_id(self) -> str:
        """Return the next synthetic loop-entry node id."""
        self._loop_entry_counter += 1
        return f"loop_entry_{self._loop_entry_counter}"

    def next_delegate_entry_id(self) -> str:
        """Return the next synthetic delegate-entry node id."""
        self._delegate_entry_counter += 1
        return f"delegate_entry_{self._delegate_entry_counter}"


def render_workflow_as_svg(steps: Sequence[WorkflowStep], *, direction: str = "TD") -> str:
    """Render one static workflow step graph as an SVG document."""
    normalized_direction = _normalize_direction(direction)
    prepared = prepare_workflow_graph(steps)
    validate_no_cycles(prepared.step_map, prepared.dependencies)
    layout = _build_svg_sequence_layout(
        builder=_SvgLayoutBuilder(),
        steps=steps,
        entry_node_id="workflow_entry",
        entry_label="Workflow Entrypoint",
        qualified_prefix=(),
        direction=normalized_direction,
    )
    return _render_svg_document(layout=layout, direction=normalized_direction)


def _build_svg_sequence_layout(
    *,
    builder: _SvgLayoutBuilder,
    steps: Sequence[WorkflowStep],
    entry_node_id: str,
    entry_label: str,
    qualified_prefix: tuple[str, ...],
    direction: str,
) -> _SvgLayout:
    """Build a deterministic SVG layout for one workflow step sequence."""
    prepared = prepare_workflow_graph(steps)
    validate_no_cycles(prepared.step_map, prepared.dependencies)
    entry_node = _svg_node(
        node_id=entry_node_id,
        label=entry_label,
        fill="#0F172A",
        stroke="#0F172A",
        text_fill="#F8FAFC",
    )
    row_layouts = _build_svg_row_layouts(
        builder=builder,
        steps=steps,
        qualified_prefix=qualified_prefix,
    )
    if direction in {"LR", "RL"}:
        return _build_svg_horizontal_sequence_layout(
            steps=steps,
            prepared_dependencies=prepared.dependencies,
            dependents=prepared.dependents,
            row_layouts=row_layouts,
            entry_node=entry_node,
            entry_node_id=entry_node_id,
        )
    return _build_svg_vertical_sequence_layout(
        steps=steps,
        prepared_dependencies=prepared.dependencies,
        dependents=prepared.dependents,
        row_layouts=row_layouts,
        entry_node=entry_node,
        entry_node_id=entry_node_id,
    )


def _build_svg_vertical_sequence_layout(
    *,
    steps: Sequence[WorkflowStep],
    prepared_dependencies: Mapping[str, tuple[str, ...]],
    dependents: Mapping[str, Sequence[str]],
    row_layouts: Sequence[_SvgRowLayout],
    entry_node: _SvgNode,
    entry_node_id: str,
) -> _SvgLayout:
    """Build a top-to-bottom SVG layout for one workflow step sequence."""
    node_column_width = max(
        [entry_node.width, *[row.step_node.width for row in row_layouts]],
        default=entry_node.width,
    )
    positioned_nodes: list[_SvgNode] = [
        _move_svg_node(
            entry_node,
            x=(node_column_width - entry_node.width) / 2.0,
            y=0.0,
        )
    ]
    positioned_edges: list[_SvgEdge] = []
    positioned_groups: list[_SvgGroup] = []
    step_node_ids: dict[str, str] = {}
    node_lookup: dict[str, _SvgNode] = {entry_node_id: positioned_nodes[0]}
    current_y = positioned_nodes[0].height + _SVG_VERTICAL_GAP

    for row in row_layouts:
        placed_node, row_height, nested_nodes, nested_edges, nested_groups = _place_svg_vertical_row(
            row=row,
            current_y=current_y,
            node_column_width=node_column_width,
        )
        positioned_nodes.append(placed_node)
        step_node_ids[row.step.step_id] = placed_node.node_id
        node_lookup[placed_node.node_id] = placed_node
        positioned_nodes.extend(nested_nodes)
        positioned_edges.extend(nested_edges)
        positioned_groups.extend(nested_groups)
        for node in nested_nodes:
            node_lookup[node.node_id] = node
        current_y += row_height + _SVG_VERTICAL_GAP

    positioned_edges.extend(
        _build_svg_sequence_edges(
            steps=steps,
            prepared_dependencies=prepared_dependencies,
            step_node_ids=step_node_ids,
            node_lookup=node_lookup,
            entry_node_id=entry_node_id,
            row_layouts=row_layouts,
        )
    )
    width, height = _svg_layout_bounds(
        nodes=positioned_nodes,
        groups=positioned_groups,
        edges=positioned_edges,
    )
    return _SvgLayout(
        nodes=tuple(positioned_nodes),
        edges=tuple(positioned_edges),
        groups=tuple(positioned_groups),
        step_node_ids=step_node_ids,
        node_lookup=node_lookup,
        entry_node_id=entry_node_id,
        terminal_node_ids=_svg_terminal_node_ids(
            steps=steps,
            step_node_ids=step_node_ids,
            dependents=dependents,
        ),
        width=width,
        height=height,
    )


def _build_svg_horizontal_sequence_layout(
    *,
    steps: Sequence[WorkflowStep],
    prepared_dependencies: Mapping[str, tuple[str, ...]],
    dependents: Mapping[str, Sequence[str]],
    row_layouts: Sequence[_SvgRowLayout],
    entry_node: _SvgNode,
    entry_node_id: str,
) -> _SvgLayout:
    """Build a left-to-right SVG layout for one workflow step sequence."""
    node_row_height = max(
        [entry_node.height, *[row.step_node.height for row in row_layouts]],
        default=entry_node.height,
    )
    positioned_entry = _move_svg_node(
        entry_node,
        x=0.0,
        y=(node_row_height - entry_node.height) / 2.0,
    )
    positioned_nodes: list[_SvgNode] = [positioned_entry]
    positioned_edges: list[_SvgEdge] = []
    positioned_groups: list[_SvgGroup] = []
    step_node_ids: dict[str, str] = {}
    node_lookup: dict[str, _SvgNode] = {entry_node_id: positioned_entry}
    current_x = positioned_entry.width + _SVG_HORIZONTAL_GAP

    for row in row_layouts:
        placed_node, column_width, nested_nodes, nested_edges, nested_groups = _place_svg_horizontal_column(
            row=row,
            current_x=current_x,
            node_row_height=node_row_height,
        )
        positioned_nodes.append(placed_node)
        step_node_ids[row.step.step_id] = placed_node.node_id
        node_lookup[placed_node.node_id] = placed_node
        positioned_nodes.extend(nested_nodes)
        positioned_edges.extend(nested_edges)
        positioned_groups.extend(nested_groups)
        for node in nested_nodes:
            node_lookup[node.node_id] = node
        current_x += column_width + _SVG_HORIZONTAL_GAP

    positioned_edges.extend(
        _build_svg_sequence_edges(
            steps=steps,
            prepared_dependencies=prepared_dependencies,
            step_node_ids=step_node_ids,
            node_lookup=node_lookup,
            entry_node_id=entry_node_id,
            row_layouts=row_layouts,
        )
    )
    width, height = _svg_layout_bounds(
        nodes=positioned_nodes,
        groups=positioned_groups,
        edges=positioned_edges,
    )
    return _SvgLayout(
        nodes=tuple(positioned_nodes),
        edges=tuple(positioned_edges),
        groups=tuple(positioned_groups),
        step_node_ids=step_node_ids,
        node_lookup=node_lookup,
        entry_node_id=entry_node_id,
        terminal_node_ids=_svg_terminal_node_ids(
            steps=steps,
            step_node_ids=step_node_ids,
            dependents=dependents,
        ),
        width=width,
        height=height,
    )


def _build_svg_row_layouts(
    *,
    builder: _SvgLayoutBuilder,
    steps: Sequence[WorkflowStep],
    qualified_prefix: tuple[str, ...],
) -> list[_SvgRowLayout]:
    """Build per-step SVG row layouts, including nested loop bodies."""
    rows: list[_SvgRowLayout] = []
    for step in steps:
        qualified_id = _qualified_step_id(qualified_prefix, step.step_id)
        step_node = _svg_step_node(
            node_id=builder.next_step_node_id(),
            step=step,
            qualified_step_id=qualified_id,
        )
        nested_layout = _build_nested_svg_layout(
            builder=builder,
            step=step,
            qualified_step_id=qualified_id,
            qualified_prefix=qualified_prefix,
        )
        rows.append(
            _SvgRowLayout(
                step=step,
                step_node=step_node,
                nested_layout=nested_layout,
            )
        )
    return rows


def _build_nested_svg_layout(
    *,
    builder: _SvgLayoutBuilder,
    step: WorkflowStep,
    qualified_step_id: str,
    qualified_prefix: tuple[str, ...],
) -> _SvgNestedLayout | None:
    """Build nested SVG layout when the step is a loop or delegate."""
    if isinstance(step, LoopStep):
        loop_entry_id = builder.next_loop_entry_id()
        return _SvgNestedLayout(
            layout=_build_svg_sequence_layout(
                builder=builder,
                steps=step.steps,
                entry_node_id=loop_entry_id,
                entry_label=f"{qualified_step_id} iteration entry",
                qualified_prefix=(*qualified_prefix, step.step_id),
                direction="TD",
            ),
            group_label=f"Loop Body: {qualified_step_id}",
            edge_label="iterate",
            edge_stroke="#7C3AED",
            terminal_edge_label="next iteration",
            terminal_edge_stroke="#7C3AED",
        )

    nested_steps = _delegate_nested_steps(step)
    if nested_steps is None:
        return None
    return _SvgNestedLayout(
        layout=_build_svg_sequence_layout(
            builder=builder,
            steps=nested_steps,
            entry_node_id=builder.next_delegate_entry_id(),
            entry_label=f"{qualified_step_id} delegate entry",
            qualified_prefix=(*qualified_prefix, step.step_id),
            direction="TD",
        ),
        group_label=f"Delegate Workflow: {qualified_step_id}",
        edge_label="delegate",
        edge_stroke="#0284C7",
    )


def _place_svg_vertical_row(
    *,
    row: _SvgRowLayout,
    current_y: float,
    node_column_width: float,
) -> tuple[_SvgNode, float, list[_SvgNode], list[_SvgEdge], list[_SvgGroup]]:
    """Position one top-to-bottom step row and any nested loop-body layout."""
    node_x = (node_column_width - row.step_node.width) / 2.0
    placed_node = _move_svg_node(row.step_node, x=node_x, y=current_y)
    if row.nested_layout is None:
        return placed_node, placed_node.height, [], [], []

    nested_x = node_column_width + _SVG_HORIZONTAL_GAP + _SVG_GROUP_PADDING
    nested_y = current_y + _SVG_GROUP_LABEL_HEIGHT
    nested_layout = _offset_svg_layout(row.nested_layout.layout, dx=nested_x, dy=nested_y)
    group = _SvgGroup(
        label=row.nested_layout.group_label,
        x=node_column_width + _SVG_HORIZONTAL_GAP,
        y=current_y,
        width=nested_layout.width + (_SVG_GROUP_PADDING * 2.0),
        height=nested_layout.height + _SVG_GROUP_LABEL_HEIGHT + _SVG_GROUP_PADDING,
    )
    terminal_edges = _build_svg_terminal_edges(
        layout=nested_layout,
        entry_node_id=row.nested_layout.layout.entry_node_id,
        edge_label=row.nested_layout.terminal_edge_label,
        stroke=row.nested_layout.terminal_edge_stroke,
    )
    return (
        placed_node,
        max(placed_node.height, group.height),
        list(nested_layout.nodes),
        [*nested_layout.edges, *terminal_edges],
        [group, *nested_layout.groups],
    )


def _place_svg_horizontal_column(
    *,
    row: _SvgRowLayout,
    current_x: float,
    node_row_height: float,
) -> tuple[_SvgNode, float, list[_SvgNode], list[_SvgEdge], list[_SvgGroup]]:
    """Position one left-to-right step column and any nested loop-body layout."""
    if row.nested_layout is None:
        placed_node = _move_svg_node(
            row.step_node,
            x=current_x,
            y=(node_row_height - row.step_node.height) / 2.0,
        )
        return placed_node, row.step_node.width, [], [], []

    group_width = row.nested_layout.layout.width + (_SVG_GROUP_PADDING * 2.0)
    group_height = row.nested_layout.layout.height + _SVG_GROUP_LABEL_HEIGHT + _SVG_GROUP_PADDING
    column_width = max(row.step_node.width, group_width)
    node_x = current_x + ((column_width - row.step_node.width) / 2.0)
    group_x = current_x + ((column_width - group_width) / 2.0)
    group_y = node_row_height + _SVG_VERTICAL_GAP
    placed_node = _move_svg_node(
        row.step_node,
        x=node_x,
        y=(node_row_height - row.step_node.height) / 2.0,
    )
    nested_layout = _offset_svg_layout(
        row.nested_layout.layout,
        dx=group_x + _SVG_GROUP_PADDING,
        dy=group_y + _SVG_GROUP_LABEL_HEIGHT,
    )
    group = _SvgGroup(
        label=row.nested_layout.group_label,
        x=group_x,
        y=group_y,
        width=group_width,
        height=group_height,
    )
    terminal_edges = _build_svg_terminal_edges(
        layout=nested_layout,
        entry_node_id=row.nested_layout.layout.entry_node_id,
        edge_label=row.nested_layout.terminal_edge_label,
        stroke=row.nested_layout.terminal_edge_stroke,
    )
    return (
        placed_node,
        column_width,
        list(nested_layout.nodes),
        [*nested_layout.edges, *terminal_edges],
        [group, *nested_layout.groups],
    )


def _build_svg_terminal_edges(
    *,
    layout: _SvgLayout,
    entry_node_id: str,
    edge_label: str | None,
    stroke: str | None,
) -> list[_SvgEdge]:
    """Build terminal edges from nested bodies back to their entry when configured."""
    if edge_label is None or stroke is None:
        return []
    entry_node = layout.node_lookup[entry_node_id]
    terminal_edges: list[_SvgEdge] = []
    for terminal_node_id in layout.terminal_node_ids:
        terminal_node = layout.node_lookup[terminal_node_id]
        terminal_edges.append(
            _svg_edge(
                source=terminal_node,
                target=entry_node,
                label=edge_label,
                dashed=True,
                stroke=stroke,
            )
        )
    return terminal_edges


def _build_svg_sequence_edges(
    *,
    steps: Sequence[WorkflowStep],
    prepared_dependencies: Mapping[str, tuple[str, ...]],
    step_node_ids: Mapping[str, str],
    node_lookup: Mapping[str, _SvgNode],
    entry_node_id: str,
    row_layouts: Sequence[_SvgRowLayout],
) -> list[_SvgEdge]:
    """Build dependency, route, and outer-loop edges for one sequence."""
    step_lookup = {row.step.step_id: row for row in row_layouts}
    workflow_step_lookup = {step.step_id: step for step in steps}
    edges: list[_SvgEdge] = []
    for step in steps:
        step_node = node_lookup[step_node_ids[step.step_id]]
        edges.extend(
            _build_svg_dependency_edges(
                dependencies=prepared_dependencies.get(step.step_id, ()),
                target_step_id=step.step_id,
                step_lookup=workflow_step_lookup,
                step_node=step_node,
                step_node_ids=step_node_ids,
                node_lookup=node_lookup,
                entry_node_id=entry_node_id,
            )
        )
        edges.extend(
            _build_svg_route_edges(
                step=step,
                step_node=step_node,
                step_node_ids=step_node_ids,
                node_lookup=node_lookup,
            )
        )
        row = step_lookup[step.step_id]
        if row.nested_layout is not None:
            nested_entry_node = node_lookup[row.nested_layout.layout.entry_node_id]
            edges.append(
                _svg_edge(
                    source=step_node,
                    target=nested_entry_node,
                    label=row.nested_layout.edge_label,
                    dashed=True,
                    stroke=row.nested_layout.edge_stroke,
                )
            )
    return edges


def _build_svg_dependency_edges(
    *,
    dependencies: Sequence[str],
    target_step_id: str,
    step_lookup: Mapping[str, WorkflowStep],
    step_node: _SvgNode,
    step_node_ids: Mapping[str, str],
    node_lookup: Mapping[str, _SvgNode],
    entry_node_id: str,
) -> list[_SvgEdge]:
    """Build solid dependency edges into one step node."""
    if not dependencies:
        return [
            _svg_edge(
                source=node_lookup[entry_node_id],
                target=step_node,
                label=None,
                dashed=False,
                stroke="#475569",
            )
        ]
    edges: list[_SvgEdge] = []
    for dependency_step_id in dependencies:
        dependency_step = step_lookup.get(dependency_step_id)
        if dependency_step is not None and _is_routed_dependency_edge(
            dependency_step=dependency_step,
            target_step_id=target_step_id,
        ):
            continue
        edges.append(
            _svg_edge(
                source=node_lookup[step_node_ids[dependency_step_id]],
                target=step_node,
                label=None,
                dashed=False,
                stroke="#475569",
            )
        )
    return edges


def _build_svg_route_edges(
    *,
    step: WorkflowStep,
    step_node: _SvgNode,
    step_node_ids: Mapping[str, str],
    node_lookup: Mapping[str, _SvgNode],
) -> list[_SvgEdge]:
    """Build dashed route edges emitted by one logic step."""
    if not isinstance(step, LogicStep) or not isinstance(step.route_map, Mapping):
        return []
    edges: list[_SvgEdge] = []
    for route_key, route_targets in step.route_map.items():
        route_label = _route_label(route_key)
        for route_target in route_targets:
            target_node_id = step_node_ids.get(route_target)
            if target_node_id is None:
                continue
            edges.append(
                _svg_edge(
                    source=step_node,
                    target=node_lookup[target_node_id],
                    label=route_label,
                    dashed=True,
                    stroke="#0F766E",
                )
            )
    return edges


def _svg_layout_bounds(
    *,
    nodes: Sequence[_SvgNode],
    groups: Sequence[_SvgGroup],
    edges: Sequence[_SvgEdge],
) -> tuple[float, float]:
    """Return concrete width/height bounds for one positioned layout."""
    max_x = max(
        [
            *[node.x + node.width for node in nodes],
            *[group.x + group.width for group in groups],
            *[x for edge in edges for x, _ in edge.points],
        ],
        default=0.0,
    )
    max_y = max(
        [
            *[node.y + node.height for node in nodes],
            *[group.y + group.height for group in groups],
            *[y for edge in edges for _, y in edge.points],
        ],
        default=0.0,
    )
    return max_x, max_y


def _is_routed_dependency_edge(*, dependency_step: WorkflowStep, target_step_id: str) -> bool:
    """Return True when one dependency is already represented as a route edge."""
    if not isinstance(dependency_step, LogicStep) or not isinstance(dependency_step.route_map, Mapping):
        return False
    return any(target_step_id in route_targets for route_targets in dependency_step.route_map.values())


def _svg_node(
    *,
    node_id: str,
    label: str,
    fill: str,
    stroke: str,
    text_fill: str,
) -> _SvgNode:
    """Build one unpositioned SVG node from a label."""
    label_lines = tuple(label.splitlines())
    longest_line = max((len(line) for line in label_lines), default=1)
    width = (longest_line * _SVG_CHARACTER_WIDTH) + (_SVG_NODE_HORIZONTAL_PADDING * 2.0)
    height = (len(label_lines) * _SVG_LINE_HEIGHT) + (_SVG_NODE_VERTICAL_PADDING * 2.0)
    return _SvgNode(
        node_id=node_id,
        label_lines=label_lines,
        x=0.0,
        y=0.0,
        width=width,
        height=height,
        fill=fill,
        stroke=stroke,
        text_fill=text_fill,
    )


def _svg_step_node(*, node_id: str, step: WorkflowStep, qualified_step_id: str) -> _SvgNode:
    """Build one unpositioned SVG node for a workflow step."""
    fill, stroke, text_fill = _svg_step_colors(step)
    return _svg_node(
        node_id=node_id,
        label=_step_label(step, qualified_step_id),
        fill=fill,
        stroke=stroke,
        text_fill=text_fill,
    )


def _svg_step_colors(step: WorkflowStep) -> tuple[str, str, str]:
    """Return deterministic SVG colors for one workflow step."""
    if isinstance(step, ToolStep):
        return "#DCFCE7", "#15803D", "#14532D"
    if isinstance(step, LoopStep):
        return "#F3E8FF", "#7E22CE", "#581C87"
    if isinstance(step, MemoryReadStep | MemoryWriteStep):
        return "#FEF3C7", "#D97706", "#78350F"
    if isinstance(step, ModelStep):
        return "#DBEAFE", "#2563EB", "#1E3A8A"
    if isinstance(step, DelegateBatchStep):
        return "#FCE7F3", "#BE185D", "#831843"
    if isinstance(step, DelegateStep):
        return "#E0F2FE", "#0284C7", "#164E63"
    return "#E2E8F0", "#475569", "#0F172A"


def _move_svg_node(node: _SvgNode, *, x: float, y: float) -> _SvgNode:
    """Return one SVG node translated to a concrete position."""
    return _SvgNode(
        node_id=node.node_id,
        label_lines=node.label_lines,
        x=x,
        y=y,
        width=node.width,
        height=node.height,
        fill=node.fill,
        stroke=node.stroke,
        text_fill=node.text_fill,
    )


def _offset_svg_layout(layout: _SvgLayout, *, dx: float, dy: float) -> _SvgLayout:
    """Translate one nested SVG layout by a fixed offset."""
    moved_nodes = tuple(_move_svg_node(node, x=node.x + dx, y=node.y + dy) for node in layout.nodes)
    moved_edges = tuple(_move_svg_edge(edge, dx=dx, dy=dy) for edge in layout.edges)
    moved_groups = tuple(
        _SvgGroup(
            label=group.label,
            x=group.x + dx,
            y=group.y + dy,
            width=group.width,
            height=group.height,
        )
        for group in layout.groups
    )
    moved_lookup = {node.node_id: node for node in moved_nodes}
    return _SvgLayout(
        nodes=moved_nodes,
        edges=moved_edges,
        groups=moved_groups,
        step_node_ids=dict(layout.step_node_ids),
        node_lookup=moved_lookup,
        entry_node_id=layout.entry_node_id,
        terminal_node_ids=layout.terminal_node_ids,
        width=layout.width,
        height=layout.height,
    )


def _move_svg_edge(edge: _SvgEdge, *, dx: float, dy: float) -> _SvgEdge:
    """Return one SVG edge translated by a fixed offset."""
    return _SvgEdge(
        points=tuple((x + dx, y + dy) for x, y in edge.points),
        label=edge.label,
        dashed=edge.dashed,
        stroke=edge.stroke,
    )


def _svg_edge(
    *,
    source: _SvgNode,
    target: _SvgNode,
    label: str | None,
    dashed: bool,
    stroke: str,
) -> _SvgEdge:
    """Build one orthogonal SVG edge between two positioned nodes."""
    return _SvgEdge(
        points=_svg_edge_points(source=source, target=target),
        label=label,
        dashed=dashed,
        stroke=stroke,
    )


def _svg_edge_points(*, source: _SvgNode, target: _SvgNode) -> tuple[tuple[float, float], ...]:
    """Return orthogonal polyline points between two nodes."""
    source_center_x = source.x + (source.width / 2.0)
    target_center_x = target.x + (target.width / 2.0)
    source_center_y = source.y + (source.height / 2.0)
    target_center_y = target.y + (target.height / 2.0)
    if target.y >= source.y + source.height:
        start = (source_center_x, source.y + source.height)
        end = (target_center_x, target.y)
        mid_y = (start[1] + end[1]) / 2.0
        return (start, (start[0], mid_y), (end[0], mid_y), end)
    if source.y >= target.y + target.height:
        lane_x = max(source.x + source.width, target.x + target.width) + 24.0
        start = (source.x + source.width, source_center_y)
        end = (target.x + target.width, target_center_y)
        return (start, (lane_x, start[1]), (lane_x, end[1]), end)
    if target.x >= source.x + source.width:
        start = (source.x + source.width, source_center_y)
        end = (target.x, target_center_y)
        mid_x = (start[0] + end[0]) / 2.0
        return (start, (mid_x, start[1]), (mid_x, end[1]), end)
    start = (source.x, source_center_y)
    end = (target.x + target.width, target_center_y)
    mid_x = (start[0] + end[0]) / 2.0
    return (start, (mid_x, start[1]), (mid_x, end[1]), end)


def _svg_terminal_node_ids(
    *,
    steps: Sequence[WorkflowStep],
    step_node_ids: Mapping[str, str],
    dependents: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Return terminal node ids for one sequence."""
    return tuple(step_node_ids[step.step_id] for step in steps if not dependents.get(step.step_id))
