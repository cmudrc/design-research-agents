"""Parser for manifest-less lazy tool docblocks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


class LazyHeaderError(ValueError):
    """Raised when lazy tool headers are malformed."""


@dataclass(slots=True, frozen=True)
class LazyInputSpec:
    """One parsed input entry from a lazy-tool header."""

    name: str
    input_type: str
    default: str | None = None


@dataclass(slots=True, frozen=True)
class LazyCapabilities:
    """Declared side-effect capabilities for one lazy tool."""

    filesystem_read: bool
    filesystem_write: bool
    network: bool
    commands: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class LazyToolHeader:
    """Normalized metadata extracted from one lazy-tool header block."""

    tool_name: str
    description: str
    inputs: tuple[LazyInputSpec, ...]
    outputs_stdout_json: bool
    capabilities: LazyCapabilities
    timeout_s: int | None
    platform: tuple[str, ...]
    version: str | None
    examples: tuple[str, ...]


_SUPPORTED_INPUT_TYPES = {
    "str",
    "int",
    "float",
    "bool",
    "path",
    "json",
    "list[str]",
    "list[int]",
}


def parse_lazy_tool_header(path: str | Path) -> LazyToolHeader:
    """Parse and validate lazy tool header directives from a script file."""
    script_path = Path(path)
    if not script_path.exists():
        raise LazyHeaderError(f"Script not found: {script_path}")

    text = script_path.read_text(encoding="utf-8")
    header_lines, first_line_no = _extract_header_block(script_path=script_path, text=text)
    return _parse_directives(
        script_path=script_path,
        lines=header_lines,
        first_line_no=first_line_no,
    )


def _extract_header_block(*, script_path: Path, text: str) -> tuple[list[str], int]:
    lines = text.splitlines()
    if script_path.suffix == ".py":
        return _extract_python_docstring(lines)
    if script_path.suffix == ".sh":
        return _extract_bash_comment_header(lines)
    raise LazyHeaderError(f"Unsupported lazy tool extension: {script_path.suffix}")


def _extract_python_docstring(lines: list[str]) -> tuple[list[str], int]:
    max_scan = min(120, len(lines))
    delimiter: str | None = None
    start_line: int | None = None

    for index in range(max_scan):
        line = lines[index]
        match = re.search(r'("""|\'\'\')', line)
        if match is None:
            continue
        delimiter = match.group(1)
        start_line = index + 1
        start_offset = match.end()
        remainder = line[start_offset:]

        if delimiter in remainder:
            end_offset = remainder.index(delimiter)
            return [remainder[:end_offset]], start_line

        block: list[str] = [remainder]
        for inner_index in range(index + 1, len(lines)):
            inner_line = lines[inner_index]
            if delimiter in inner_line:
                end_pos = inner_line.index(delimiter)
                block.append(inner_line[:end_pos])
                return block, start_line
            block.append(inner_line)

        raise LazyHeaderError("Unterminated python docstring header.")

    raise LazyHeaderError(
        "Missing header docstring in first 120 lines. "
        "Add a triple-quoted header block at the top of the file."
    )


def _extract_bash_comment_header(lines: list[str]) -> tuple[list[str], int]:
    max_scan = min(120, len(lines))
    index = 0
    if lines and lines[0].startswith("#!"):
        index = 1

    block: list[str] = []
    first_line_no: int | None = None
    while index < max_scan:
        line = lines[index]
        stripped = line.lstrip()
        if not stripped.startswith("#"):
            break
        if first_line_no is None:
            first_line_no = index + 1
        content = stripped[1:]
        if content.startswith(" "):
            content = content[1:]
        block.append(content)
        index += 1

    if not block:
        raise LazyHeaderError(
            "Missing bash header comment block in first 120 lines. Add leading '#' directives."
        )

    return block, first_line_no or 1


def _parse_directives(*, script_path: Path, lines: list[str], first_line_no: int) -> LazyToolHeader:
    key_values: dict[str, str] = {}
    sections: dict[str, list[tuple[int, str]]] = {}
    current_section: str | None = None
    seen_directive = False

    for offset, raw_line in enumerate(lines):
        line_number = first_line_no + offset
        line = raw_line.rstrip()
        if not line.strip():
            continue

        if line.lstrip().startswith("@"):
            seen_directive = True
            current_section = None
            match = re.match(r"^\s*@([a-z_]+):\s*(.*)$", line)
            if match is None:
                raise LazyHeaderError(f"Invalid directive line (line {line_number}).")
            key = match.group(1)
            value = match.group(2)
            if key in {"inputs", "outputs", "capabilities", "examples"}:
                sections[key] = []
                current_section = key
                if value.strip():
                    sections[key].append((line_number, value.strip()))
                continue
            key_values[key] = value.strip()
            continue

        if not seen_directive:
            continue

        if current_section is None:
            raise LazyHeaderError(
                f"Unexpected non-directive content (line {line_number}). "
                "Start lines with @key: or indent section members."
            )
        sections.setdefault(current_section, []).append((line_number, line.strip()))

    if not key_values.get("tool_name"):
        raise LazyHeaderError("Missing @tool_name in header (required). Add: @tool_name: my_tool")
    if not key_values.get("description"):
        raise LazyHeaderError("Missing @description in header (required). Add: @description: ...")

    tool_name = key_values["tool_name"]
    if not re.match(r"^[a-z][a-z0-9_]*$", tool_name):
        raise LazyHeaderError("@tool_name must be snake_case.")

    inputs = _parse_inputs(sections.get("inputs", []))
    outputs_stdout_json = _parse_outputs(sections.get("outputs", []))
    capabilities = _parse_capabilities(sections.get("capabilities", []))

    timeout_s: int | None = None
    if "timeout_s" in key_values:
        try:
            timeout_s = int(key_values["timeout_s"])
        except ValueError as exc:
            raise LazyHeaderError("@timeout_s must be an integer.") from exc

    platform = _parse_platform(key_values.get("platform"))
    version = key_values.get("version") or None
    examples = _parse_examples(sections.get("examples", []))

    return LazyToolHeader(
        tool_name=tool_name,
        description=key_values["description"],
        inputs=tuple(inputs),
        outputs_stdout_json=outputs_stdout_json,
        capabilities=capabilities,
        timeout_s=timeout_s,
        platform=platform,
        version=version,
        examples=examples,
    )


def _parse_inputs(lines: list[tuple[int, str]]) -> list[LazyInputSpec]:
    if not lines:
        return []
    parsed: list[LazyInputSpec] = []
    for line_number, line in lines:
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*([^=]+?)(?:\s*=\s*(.+))?$", line)
        if match is None:
            raise LazyHeaderError(
                f"Invalid input line (line {line_number}): expected '<name>: <type> [= default]'"
            )
        name = match.group(1)
        input_type = match.group(2).strip()
        default = match.group(3).strip() if match.group(3) is not None else None
        if not _is_supported_input_type(input_type):
            raise LazyHeaderError(f"Unsupported input type '{input_type}' (line {line_number}).")
        parsed.append(LazyInputSpec(name=name, input_type=input_type, default=default))
    return parsed


def _is_supported_input_type(input_type: str) -> bool:
    if input_type in _SUPPORTED_INPUT_TYPES:
        return True
    return bool(re.match(r"^enum\[[^\]]+\]$", input_type))


def _parse_outputs(lines: list[tuple[int, str]]) -> bool:
    if not lines:
        raise LazyHeaderError("Missing @outputs block (required).")
    output_map: dict[str, str] = {}
    for line_number, line in lines:
        match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.+)$", line)
        if match is None:
            raise LazyHeaderError(f"Invalid @outputs line (line {line_number}).")
        output_map[match.group(1)] = match.group(2).strip().lower()
    stdout_json = output_map.get("stdout_json")
    if stdout_json != "true":
        raise LazyHeaderError("outputs must include stdout_json: true")
    return True


def _parse_capabilities(lines: list[tuple[int, str]]) -> LazyCapabilities:
    if not lines:
        raise LazyHeaderError("Missing @capabilities block (required).")

    values: dict[str, str] = {}
    for line_number, line in lines:
        match = re.match(r"^([a-z_]+)\s*:\s*(.+)$", line)
        if match is None:
            raise LazyHeaderError(f"Invalid @capabilities line (line {line_number}).")
        values[match.group(1)] = match.group(2).strip()

    def _bool_field(name: str) -> bool:
        raw = values.get(name)
        if raw is None:
            raise LazyHeaderError(f"Missing capability '{name}' in @capabilities block.")
        normalized = raw.lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        raise LazyHeaderError(f"Capability '{name}' must be true/false.")

    commands_raw = values.get("commands", "[]")
    commands = _parse_list_literal(commands_raw)

    return LazyCapabilities(
        filesystem_read=_bool_field("filesystem_read"),
        filesystem_write=_bool_field("filesystem_write"),
        network=_bool_field("network"),
        commands=tuple(commands),
    )


def _parse_platform(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ()
    values = _parse_list_literal(raw)
    for value in values:
        if value not in {"darwin", "linux", "windows"}:
            raise LazyHeaderError(f"Unsupported platform '{value}'.")
    return tuple(values)


def _parse_examples(lines: list[tuple[int, str]]) -> tuple[str, ...]:
    if not lines:
        return ()
    parsed: list[str] = []
    for _, line in lines:
        normalized = line
        if normalized.startswith("-"):
            normalized = normalized[1:].strip()
        if normalized:
            parsed.append(normalized)
    return tuple(parsed)


def _parse_list_literal(raw: str) -> list[str]:
    normalized = raw.strip()
    if not normalized.startswith("[") or not normalized.endswith("]"):
        raise LazyHeaderError(f"Expected list literal like [a,b], got: {raw!r}")
    body = normalized[1:-1].strip()
    if not body:
        return []
    items = [item.strip() for item in body.split(",")]
    return [item for item in items if item]


__all__ = [
    "LazyCapabilities",
    "LazyHeaderError",
    "LazyInputSpec",
    "LazyToolHeader",
    "parse_lazy_tool_header",
]
