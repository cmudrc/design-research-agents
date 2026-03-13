"""SVG rendering helpers for workflow diagram layouts."""

from __future__ import annotations

from html import escape as html_escape

from ._diagram_svg_types import (
    _SVG_LINE_HEIGHT,
    _SVG_NODE_VERTICAL_PADDING,
    _SvgEdge,
    _SvgGroup,
    _SvgLayout,
    _SvgNode,
)

_SVG_CANVAS_PADDING = 24.0


def _render_svg_document(*, layout: _SvgLayout, direction: str) -> str:
    """Render one SVG layout into a standalone SVG document."""
    canvas_width, canvas_height = _svg_canvas_size(width=layout.width, height=layout.height, direction=direction)
    rendered_groups = "\n".join(
        _render_svg_group(
            group=_transform_svg_group(
                group=group,
                width=layout.width,
                height=layout.height,
                direction=direction,
            )
        )
        for group in layout.groups
    )
    rendered_edges = "\n".join(
        _render_svg_edge(
            edge=_transform_svg_edge(
                edge=edge,
                width=layout.width,
                height=layout.height,
                direction=direction,
            )
        )
        for edge in layout.edges
    )
    rendered_nodes = "\n".join(
        _render_svg_node(
            node=_transform_svg_node(
                node=node,
                width=layout.width,
                height=layout.height,
                direction=direction,
            )
        )
        for node in layout.nodes
    )
    return "\n".join(
        [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width + (_SVG_CANVAS_PADDING * 2.0):.0f}" '
                f'height="{canvas_height + (_SVG_CANVAS_PADDING * 2.0):.0f}" '
                f'viewBox="0 0 {canvas_width + (_SVG_CANVAS_PADDING * 2.0):.0f} '
                f'{canvas_height + (_SVG_CANVAS_PADDING * 2.0):.0f}" role="img" '
                'aria-label="Workflow diagram">'
            ),
            "  <defs>",
            (
                '    <marker id="workflow-arrow" viewBox="0 0 10 10" '
                'refX="8" refY="5" markerWidth="7" markerHeight="7" '
                'orient="auto-start-reverse">'
            ),
            '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#475569" />',
            "    </marker>",
            "  </defs>",
            '  <rect x="0" y="0" width="100%" height="100%" fill="#FFFFFF" />',
            f'  <g transform="translate({_SVG_CANVAS_PADDING:.0f},{_SVG_CANVAS_PADDING:.0f})">',
            rendered_groups,
            rendered_edges,
            rendered_nodes,
            "  </g>",
            "</svg>",
        ]
    )


def _svg_canvas_size(*, width: float, height: float, direction: str) -> tuple[float, float]:
    """Return canvas dimensions for one layout direction."""
    if direction in {"TD", "BT"}:
        return width, height
    return height, width


def _transform_svg_node(*, node: _SvgNode, width: float, height: float, direction: str) -> _SvgNode:
    """Transform one SVG node into the requested output direction."""
    if direction == "TD":
        return node
    if direction == "BT":
        return _SvgNode(
            node_id=node.node_id,
            label_lines=node.label_lines,
            x=node.x,
            y=height - node.y - node.height,
            width=node.width,
            height=node.height,
            fill=node.fill,
            stroke=node.stroke,
            text_fill=node.text_fill,
        )
    if direction == "LR":
        return _SvgNode(
            node_id=node.node_id,
            label_lines=node.label_lines,
            x=node.y,
            y=node.x,
            width=node.width,
            height=node.height,
            fill=node.fill,
            stroke=node.stroke,
            text_fill=node.text_fill,
        )
    return _SvgNode(
        node_id=node.node_id,
        label_lines=node.label_lines,
        x=height - node.y - node.width,
        y=node.x,
        width=node.width,
        height=node.height,
        fill=node.fill,
        stroke=node.stroke,
        text_fill=node.text_fill,
    )


def _transform_svg_group(*, group: _SvgGroup, width: float, height: float, direction: str) -> _SvgGroup:
    """Transform one SVG group box into the requested output direction."""
    del width
    if direction == "TD":
        return group
    if direction == "BT":
        return _SvgGroup(
            label=group.label,
            x=group.x,
            y=height - group.y - group.height,
            width=group.width,
            height=group.height,
        )
    if direction == "LR":
        return _SvgGroup(
            label=group.label,
            x=group.y,
            y=group.x,
            width=group.height,
            height=group.width,
        )
    return _SvgGroup(
        label=group.label,
        x=height - group.y - group.height,
        y=group.x,
        width=group.height,
        height=group.width,
    )


def _transform_svg_edge(*, edge: _SvgEdge, width: float, height: float, direction: str) -> _SvgEdge:
    """Transform one SVG edge into the requested output direction."""
    return _SvgEdge(
        points=tuple(
            _transform_svg_point(x=x, y=y, width=width, height=height, direction=direction) for x, y in edge.points
        ),
        label=edge.label,
        dashed=edge.dashed,
        stroke=edge.stroke,
    )


def _transform_svg_point(*, x: float, y: float, width: float, height: float, direction: str) -> tuple[float, float]:
    """Transform one SVG point into the requested output direction."""
    del width
    if direction == "TD":
        return x, y
    if direction == "BT":
        return x, height - y
    if direction == "LR":
        return y, x
    return height - y, x


def _render_svg_group(*, group: _SvgGroup) -> str:
    """Render one SVG loop-body group rectangle."""
    label_y = group.y + 20.0
    return "\n".join(
        [
            (
                f'    <rect x="{group.x:.1f}" y="{group.y:.1f}" width="{group.width:.1f}" '
                f'height="{group.height:.1f}" rx="14" ry="14" fill="#FAFAFF" '
                'stroke="#C4B5FD" stroke-width="1.5" />'
            ),
            (
                f'    <text x="{group.x + 14.0:.1f}" y="{label_y:.1f}" '
                'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
                'font-size="13" font-weight="600" fill="#6D28D9">'
                f"{html_escape(group.label)}</text>"
            ),
        ]
    )


def _render_svg_edge(*, edge: _SvgEdge) -> str:
    """Render one SVG edge polyline and optional label."""
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in edge.points)
    dash_array = ' stroke-dasharray="7 5"' if edge.dashed else ""
    label_block = _render_svg_edge_label(edge=edge)
    return "\n".join(
        [
            (
                f'    <polyline points="{points}" fill="none" stroke="{edge.stroke}" '
                f'stroke-width="2.0"{dash_array} marker-end="url(#workflow-arrow)" '
                'stroke-linecap="round" stroke-linejoin="round" />'
            ),
            label_block,
        ]
    ).strip()


def _render_svg_edge_label(*, edge: _SvgEdge) -> str:
    """Render one optional SVG edge label."""
    if not edge.label:
        return ""
    label_x, label_y = edge.points[max(0, len(edge.points) // 2 - 1)]
    return (
        f'    <text x="{label_x + 6.0:.1f}" y="{label_y - 6.0:.1f}" '
        'font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
        'font-size="12" fill="#334155" '
        'paint-order="stroke" stroke="#FFFFFF" stroke-width="4">'
        f"{html_escape(edge.label)}</text>"
    )


def _render_svg_node(*, node: _SvgNode) -> str:
    """Render one SVG node rectangle and multiline label."""
    text_x = node.x + (node.width / 2.0)
    first_line_y = node.y + _SVG_NODE_VERTICAL_PADDING + 12.0
    tspans = "\n".join(
        (
            f'      <tspan x="{text_x:.1f}" dy="0">{html_escape(node.label_lines[0])}</tspan>'
            if index == 0
            else f'      <tspan x="{text_x:.1f}" dy="{_SVG_LINE_HEIGHT:.1f}">{html_escape(line)}</tspan>'
        )
        for index, line in enumerate(node.label_lines)
    )
    return "\n".join(
        [
            (
                f'    <rect id="{node.node_id}" x="{node.x:.1f}" y="{node.y:.1f}" '
                f'width="{node.width:.1f}" height="{node.height:.1f}" rx="12" ry="12" '
                f'fill="{node.fill}" stroke="{node.stroke}" stroke-width="1.6" />'
            ),
            (
                f'    <text x="{text_x:.1f}" y="{first_line_y:.1f}" '
                'text-anchor="middle" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" '
                f'font-size="13" fill="{node.text_fill}">'
            ),
            tspans,
            "    </text>",
        ]
    )
