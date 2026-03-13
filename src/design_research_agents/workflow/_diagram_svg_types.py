"""Shared SVG datatypes and constants for workflow diagram rendering."""

from __future__ import annotations

from dataclasses import dataclass

from design_research_agents._contracts._workflow import WorkflowStep

_SVG_NODE_VERTICAL_PADDING = 12.0
_SVG_LINE_HEIGHT = 18.0


@dataclass(slots=True, frozen=True)
class _SvgNode:
    """One positioned SVG node box."""

    node_id: str
    label_lines: tuple[str, ...]
    x: float
    y: float
    width: float
    height: float
    fill: str
    stroke: str
    text_fill: str


@dataclass(slots=True, frozen=True)
class _SvgEdge:
    """One positioned SVG edge polyline."""

    points: tuple[tuple[float, float], ...]
    label: str | None
    dashed: bool
    stroke: str


@dataclass(slots=True, frozen=True)
class _SvgGroup:
    """One positioned SVG loop-body group box."""

    label: str
    x: float
    y: float
    width: float
    height: float


@dataclass(slots=True, frozen=True)
class _SvgRowLayout:
    """Precomputed row layout for one step and any nested loop body."""

    step: WorkflowStep
    step_node: _SvgNode
    nested_layout: _SvgLayout | None


@dataclass(slots=True, frozen=True)
class _SvgLayout:
    """Rendered SVG layout payload for one workflow sequence."""

    nodes: tuple[_SvgNode, ...]
    edges: tuple[_SvgEdge, ...]
    groups: tuple[_SvgGroup, ...]
    step_node_ids: dict[str, str]
    node_lookup: dict[str, _SvgNode]
    entry_node_id: str
    terminal_node_ids: tuple[str, ...]
    width: float
    height: float
