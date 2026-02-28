"""Parsing and normalization helpers for code-writing tool-calling agents."""

from __future__ import annotations

import ast
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from design_research_agents._contracts._tools import ToolSpec


@dataclass(slots=True, frozen=True)
class AllowedTool:
    """Normalized allowed-tool definition used during one run."""

    tool_name: str
    """Canonical tool name exposed to generated code."""
    description: str
    """Human-readable tool description shown in prompts."""
    input_schema: dict[str, object]
    """Normalized input schema used for validation."""
    default_tool_input: dict[str, object] | None = None
    """Optional default input merged in when generated code passes an empty object."""


@dataclass(slots=True, frozen=True)
class CodeNormalizationResult:
    """Captures optional pre-validation code normalization details."""

    code_text: str
    """Normalized code text after safe rewrites."""
    raw_code_text: str
    """Original code text before normalization."""
    stripped_safe_tool_imports: int
    """Count of removable safe tool imports that were stripped."""
    rewritten_tool_calls: int
    """Count of tool-call expressions rewritten into canonical form."""
    rewritten_direct_name_calls: int
    """Count of direct-name tool calls rewritten."""
    rewritten_module_attr_calls: int
    """Count of module-attribute tool calls rewritten."""
    parse_error: str | None = None
    """Parse error captured when normalization could not inspect the code."""

    @property
    def changed(self) -> bool:
        """Return whether normalization altered imports/calls.

        Returns:
            ``True`` when code was rewritten or safe imports were stripped.
        """
        return self.stripped_safe_tool_imports > 0 or self.rewritten_tool_calls > 0

    def metadata(self) -> dict[str, object]:
        """Return a compact normalization metadata payload.

        Returns:
            Serializable metadata describing the normalization pass.
        """
        return {
            "changed": self.changed,
            "stripped_safe_tool_imports": self.stripped_safe_tool_imports,
            "rewritten_tool_calls": self.rewritten_tool_calls,
            "rewritten_direct_name_calls": self.rewritten_direct_name_calls,
            "rewritten_module_attr_calls": self.rewritten_module_attr_calls,
            "parse_error": self.parse_error,
        }


def extract_allowed_tools(
    *,
    default_allowed_tools: Sequence[AllowedTool],
) -> tuple[list[AllowedTool], str]:
    """Return allowed tools compiled at initialization time.

    Args:
        default_allowed_tools: Run-level default tool definitions.

    Returns:
        Cloned allowed-tool list plus a source label for metadata.
    """
    return (
        [clone_allowed_tool(tool) for tool in default_allowed_tools],
        "init_default",
    )


def compile_default_allowed_tools(
    *,
    runtime_specs: Mapping[str, ToolSpec],
    default_tools: Sequence[Mapping[str, object]] | None,
) -> tuple[AllowedTool, ...]:
    """Compile default allowed tools from init config and runtime tool specs.

    Args:
        runtime_specs: Runtime tool specs keyed by tool name.
        default_tools: Optional explicit allowlist payload supplied at initialization.

    Returns:
        Immutable allowed-tool tuple for future runs.
    """
    if default_tools is not None:
        compiled_from_input = normalize_allowed_tools(
            raw_tools=default_tools,
            runtime_specs=runtime_specs,
        )
        return tuple(compiled_from_input)

    return tuple(
        AllowedTool(
            tool_name=spec.name,
            description=spec.description,
            input_schema=dict(spec.input_schema),
        )
        for spec in runtime_specs.values()
    )


def normalize_allowed_tools(
    *,
    raw_tools: object,
    runtime_specs: Mapping[str, ToolSpec],
) -> list[AllowedTool]:
    """Normalize explicit allowed-tool payload into runtime-backed tool entries.

    Args:
        raw_tools: Raw allowlist payload to normalize.
        runtime_specs: Runtime tool specs keyed by tool name.

    Returns:
        Normalized allowed tools that map to registered runtime tools.
    """
    if not isinstance(raw_tools, Sequence) or isinstance(raw_tools, (str, bytes)):
        return []

    normalized: list[AllowedTool] = []
    for raw_tool in raw_tools:
        if not isinstance(raw_tool, Mapping):
            continue
        raw_tool_name = raw_tool.get("tool_name", raw_tool.get("name"))
        if not isinstance(raw_tool_name, str):
            continue
        tool_name = raw_tool_name.strip()
        if not tool_name:
            continue

        runtime_spec = runtime_specs.get(tool_name)
        if runtime_spec is None:
            # Only runtime-registered tools can be called.
            continue

        raw_description = raw_tool.get("description")
        description = str(raw_description) if raw_description is not None else runtime_spec.description

        raw_input_schema = raw_tool.get("input_schema")
        if isinstance(raw_input_schema, Mapping):
            input_schema = dict(raw_input_schema)
        else:
            input_schema = dict(runtime_spec.input_schema)

        raw_default_tool_input = raw_tool.get("tool_input")
        default_tool_input = dict(raw_default_tool_input) if isinstance(raw_default_tool_input, Mapping) else None
        normalized.append(
            AllowedTool(
                tool_name=tool_name,
                description=description,
                input_schema=input_schema,
                default_tool_input=default_tool_input,
            )
        )

    deduped: dict[str, AllowedTool] = {}
    for allowed_tool in normalized:
        deduped[allowed_tool.tool_name] = allowed_tool
    return list(deduped.values())


def clone_allowed_tool(allowed_tool: AllowedTool) -> AllowedTool:
    """Clone one allowed tool to isolate run-level payload mutations.

    Args:
        allowed_tool: Allowed-tool definition to clone.

    Returns:
        Deep-ish copy safe for run-level mutations.
    """
    return AllowedTool(
        tool_name=allowed_tool.tool_name,
        description=allowed_tool.description,
        input_schema=dict(allowed_tool.input_schema),
        default_tool_input=(
            dict(allowed_tool.default_tool_input) if isinstance(allowed_tool.default_tool_input, Mapping) else None
        ),
    )


def extract_prompt(input_payload: Mapping[str, object]) -> str:
    """Extract prompt text from run input.

    Args:
        input_payload: Run input payload that may contain prompt text.

    Returns:
        Prompt text to send to the model.
    """
    raw_prompt = input_payload.get("prompt", input_payload.get("text", "Provide a concise response."))
    return str(raw_prompt)


def extract_positive_int(
    *,
    input_payload: Mapping[str, object],
    key: str,
    default_value: int,
) -> int:
    """Extract a positive integer option from run input.

    Args:
        input_payload: Run input payload that may contain the option.
        key: Option name to read from ``input_payload``.
        default_value: Fallback value when the option is missing or invalid.

    Returns:
        Positive integer value resolved from input or fallback.
    """
    raw_value = input_payload.get(key)
    if raw_value is None:
        return default_value
    if isinstance(raw_value, bool):
        return default_value
    if isinstance(raw_value, int) and raw_value >= 1:
        return raw_value
    return default_value


def extract_boolean(
    *,
    input_payload: Mapping[str, object],
    key: str,
    default_value: bool,
) -> bool:
    """Extract a boolean option from run input.

    Args:
        input_payload: Run input payload that may contain the option.
        key: Option name to read from ``input_payload``.
        default_value: Fallback value when the option is missing or invalid.

    Returns:
        Boolean value resolved from input or fallback.
    """
    raw_value = input_payload.get(key)
    if isinstance(raw_value, bool):
        return raw_value
    return default_value


def extract_python_code(raw_model_text: str) -> str:
    """Extract Python code from model output text, preferring fenced blocks.

    Args:
        raw_model_text: Raw text emitted by the model.

    Returns:
        Candidate Python code text for compilation.
    """
    fenced_match = match_fenced_code_block(raw_model_text)
    if fenced_match is not None:
        return fenced_match.strip()
    return raw_model_text.strip()


def match_fenced_code_block(raw_text: str) -> str | None:
    """Return first Python-like fenced code block when present and well-formed.

    Args:
        raw_text: Raw model text to inspect.

    Returns:
        First Python-like fenced code block, or ``None`` when none is found.
    """
    fence = "```"
    start_index = raw_text.find(fence)
    if start_index == -1:
        return None
    end_of_fence_header = raw_text.find("\n", start_index + len(fence))
    if end_of_fence_header == -1:
        return None

    fence_header = raw_text[start_index + len(fence) : end_of_fence_header].strip().lower()
    end_index = raw_text.find(fence, end_of_fence_header + 1)
    if end_index == -1:
        return None

    if fence_header not in {"", "python", "py"}:
        # Only consume the block when header is absent or python-like.
        return None
    return raw_text[end_of_fence_header + 1 : end_index]


class _CodeCanonicalizer(ast.NodeTransformer):
    """Conservative AST normalizer for common tool-call variants."""

    def __init__(self, *, allowed_tool_names: set[str]) -> None:
        """Initialize the canonicalizer with the set of allowed tool names.

        Args:
            allowed_tool_names: Tool names that may be rewritten into ``call_tool``.
        """
        self._allowed_tool_names = allowed_tool_names
        self.stripped_safe_tool_imports = 0
        self.rewritten_tool_calls = 0
        self.rewritten_direct_name_calls = 0
        self.rewritten_module_attr_calls = 0

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Strip safe imports and rewrite calls across the module body.

        Args:
            node: Module node being normalized.

        Returns:
            Rewritten module node.
        """
        new_body: list[ast.stmt] = []
        for statement in node.body:
            if self._is_strippable_tool_import(statement):
                self.stripped_safe_tool_imports += 1
                continue
            visited = self.visit(statement)
            if visited is None:
                continue
            if isinstance(visited, list):
                new_body.extend(visited)
                continue
            new_body.append(visited)
        node.body = new_body
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Rewrite supported tool-call forms into canonical ``call_tool`` calls.

        Args:
            node: Call node being normalized.

        Returns:
            Rewritten call node, or the original node when no rewrite applies.
        """
        visited_node = self.generic_visit(node)
        if not isinstance(visited_node, ast.Call):
            return visited_node

        target = self._resolve_rewrite_target(visited_node.func)
        if target is None:
            return visited_node

        tool_name, call_style = target
        tool_input_arg = self._build_tool_input_argument(visited_node)
        if tool_input_arg is None:
            return visited_node

        rewritten_call = ast.Call(
            func=ast.Name(id="call_tool", ctx=ast.Load()),
            args=[ast.Constant(value=tool_name), tool_input_arg],
            keywords=[],
        )
        rewritten_call = ast.copy_location(rewritten_call, visited_node)
        self.rewritten_tool_calls += 1
        if call_style == "direct_name":
            self.rewritten_direct_name_calls += 1
        else:
            self.rewritten_module_attr_calls += 1
        return rewritten_call

    def _is_strippable_tool_import(self, statement: ast.stmt) -> bool:
        """Execute is strippable tool import.

        Args:
            statement: Value supplied for ``statement``.

        Returns:
            Result produced by this call.
        """
        return isinstance(statement, (ast.Import, ast.ImportFrom))

    def _resolve_rewrite_target(
        self,
        call_target: ast.expr,
    ) -> tuple[str, str] | None:
        """Execute resolve rewrite target.

        Args:
            call_target: Value supplied for ``call_target``.

        Returns:
            Result produced by this call.
        """
        if isinstance(call_target, ast.Name):
            if call_target.id in self._allowed_tool_names:
                return call_target.id, "direct_name"
            return None
        if not isinstance(call_target, ast.Attribute):
            return None
        if not isinstance(call_target.value, ast.Name):
            return None
        if call_target.value.id != call_target.attr:
            return None
        if call_target.attr not in self._allowed_tool_names:
            return None
        return call_target.attr, "module_attr"

    def _build_tool_input_argument(self, node: ast.Call) -> ast.expr | None:
        """Execute build tool input argument.

        Args:
            node: Value supplied for ``node``.

        Returns:
            Result produced by this call.
        """
        if len(node.args) == 1 and not node.keywords:
            if isinstance(node.args[0], ast.Starred):
                return None
            return node.args[0]
        if node.args:
            return None
        if not node.keywords:
            return ast.Dict(keys=[], values=[])

        keys: list[ast.expr | None] = []
        values: list[ast.expr] = []
        for keyword in node.keywords:
            if keyword.arg is None:
                return None
            keys.append(ast.Constant(value=keyword.arg))
            values.append(keyword.value)
        return ast.Dict(keys=keys, values=values)


def canonicalize_generated_code(
    *,
    code_text: str,
    allowed_tools: Sequence[AllowedTool],
) -> CodeNormalizationResult:
    """Rewrite narrow, known-safe tool call variants into canonical form.

    Args:
        code_text: Value supplied for ``code_text``.
        allowed_tools: Value supplied for ``allowed_tools``.

    Returns:
        Result produced by this call.
    """
    if not code_text:
        return CodeNormalizationResult(
            code_text=code_text,
            raw_code_text=code_text,
            stripped_safe_tool_imports=0,
            rewritten_tool_calls=0,
            rewritten_direct_name_calls=0,
            rewritten_module_attr_calls=0,
            parse_error=None,
        )

    try:
        syntax_tree = ast.parse(code_text, mode="exec")
    except SyntaxError as exc:
        return CodeNormalizationResult(
            code_text=code_text,
            raw_code_text=code_text,
            stripped_safe_tool_imports=0,
            rewritten_tool_calls=0,
            rewritten_direct_name_calls=0,
            rewritten_module_attr_calls=0,
            parse_error=str(exc),
        )

    canonicalizer = _CodeCanonicalizer(allowed_tool_names={tool.tool_name for tool in allowed_tools})
    normalized_tree = canonicalizer.visit(syntax_tree)
    ast.fix_missing_locations(normalized_tree)

    if canonicalizer.stripped_safe_tool_imports == 0 and canonicalizer.rewritten_tool_calls == 0:
        normalized_code = code_text
    else:
        normalized_code = ast.unparse(normalized_tree).strip()

    return CodeNormalizationResult(
        code_text=normalized_code,
        raw_code_text=code_text,
        stripped_safe_tool_imports=canonicalizer.stripped_safe_tool_imports,
        rewritten_tool_calls=canonicalizer.rewritten_tool_calls,
        rewritten_direct_name_calls=canonicalizer.rewritten_direct_name_calls,
        rewritten_module_attr_calls=canonicalizer.rewritten_module_attr_calls,
        parse_error=None,
    )


__all__ = [
    "AllowedTool",
    "CodeNormalizationResult",
    "canonicalize_generated_code",
    "clone_allowed_tool",
    "compile_default_allowed_tools",
    "extract_allowed_tools",
    "extract_boolean",
    "extract_positive_int",
    "extract_prompt",
    "extract_python_code",
]
