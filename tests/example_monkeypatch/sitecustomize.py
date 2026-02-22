"""Test-only monkeypatches for running examples with deterministic LLM responses.

Loaded automatically by Python when this directory is prepended to PYTHONPATH.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

_DETERMINISTIC_MODE = os.environ.get("DRA_EXAMPLE_LLM_MODE", "").strip().lower() == "deterministic"

if _DETERMINISTIC_MODE:
    import design_research_agents.llm as llm_module
    from design_research_agents._contracts._llm import (
        LLMChatParams,
        LLMDelta,
        LLMMessage,
        LLMRequest,
        LLMResponse,
        LLMStreamEvent,
    )

_SCRIPT_RESPONSES: dict[str, tuple[str, ...]] = {
    "examples/agents/basic/direct_llm_call.py": ("4",),
    "examples/agents/basic/multi_step_code_tool_calling_agent.py": (
        '{"continue": true, "thought": "Read README and analyze text stats."}',
        "\n".join(
            [
                'readme = call_tool("fs.read_text", {"path": "README.md", "max_bytes": 2400})',
                'stats = call_tool("text.word_count", {"text": readme["text"]})',
                'diff_result = call_tool("text.diff", {"a": "core tools only", "b": "core + script + mcp tools"})',
                "final_output = {",
                '    "word_count": stats["word_count"],',
                '    "line_count": stats["line_count"],',
                '    "diff_preview": diff_result["diff"].splitlines()[:4],',
                "}",
            ]
        ),
        '{"continue": false, "thought": "Task complete."}',
    ),
    "examples/agents/basic/multi_step_json_tool_calling_agent.py": (
        '{"continue": true, "thought": "Read README first."}',
        (
            '{"tool_name":"fs.read_text","tool_input":{"path":"README.md","max_bytes":800},'
            '"reason":"Need repository text context."}'
        ),
        '{"continue": false, "thought": "Task complete."}',
    ),
    "examples/agents/basic/multi_step_json_with_memory.py": (
        '{"continue": true, "thought": "start"}',
        '{"tool_name": "text.word_count", "tool_input": {"text": "design context memory"}}',
        '{"continue": false, "thought": "done"}',
    ),
    "examples/optimization/multi_step_tool_router_1d_optimization.py": (
        '{"continue": true, "thought": "Start descending toward zero."}',
        ('{"tool_name":"optimizer.decrease_x","tool_input":{"step":1},"reason":"Decrease x toward zero."}'),
        '{"continue": true, "thought": "Still above zero, continue decreasing."}',
        ('{"tool_name":"optimizer.decrease_x","tool_input":{"step":1},"reason":"Keep moving toward zero."}'),
        '{"continue": true, "thought": "One more decrease should reach zero."}',
        ('{"tool_name":"optimizer.decrease_x","tool_input":{"step":1},"reason":"Reach x=0."}'),
        '{"continue": false, "thought": "No better one-step move remains."}',
    ),
    "examples/agents/basic/multi_step_direct_llm_agent.py": (
        ('{"decision":"CONTINUE","content":"Draft answer: compute 6 * 7.","reason":"Need final wording."}'),
        '{"decision":"STOP","content":"Final answer ready.","final_output":"42","reason":"done"}',
    ),
    "examples/workflow/plan_execute.py": (
        (
            '{"steps":[{"step_id":"analyze_repo_tools","instruction":"Write a small CSV artifact '
            "describing tool sources, then describe that CSV and search the tools package for "
            'Toolbox.","success_criteria":"Return csv row stats and source-code '
            'match count."}]}'
        ),
        '{"continue": true, "thought": "Run the first execution step."}',
        "\n".join(
            [
                'csv_text = "tool,source\\ntext.word_count,core\\nrubric_score,script\\ntext.word_count,mcp\\n"',
                'write_result = call_tool("fs.write_text", {"path": '
                '"artifacts/examples/plan_execute_runtime_inventory.csv", "content": csv_text, '
                '"overwrite": True})',
                'describe_result = call_tool("data.describe", {"path": '
                '"artifacts/examples/plan_execute_runtime_inventory.csv", "kind": "csv"})',
                'search_result = call_tool("search.ripgrep", {"query": '
                '"Toolbox", "root": "src/design_research_agents/tools", '
                '"max_matches": 4})',
                "final_output = {",
                '  "csv_path": write_result["path"],',
                '  "row_count": describe_result["rows"],',
                '  "column_count": describe_result["column_count"],',
                '  "search_hits": search_result["count"],',
                "}",
            ]
        ),
    ),
    "examples/workflow/propose_critic.py": (
        "Draft v1: simple proposal.",
        ('{"approved": false, "feedback": "Add more detail.", "revision_goals": ["expand rationale"]}'),
        "Draft v2: proposal with more detail.",
        '{"approved": true, "feedback": "Looks good.", "revision_goals": []}',
    ),
    "examples/workflow/agent_routing.py": (
        ('{"action":"TOOL_CALL","tool_names":["json_tool_agent"],"reason":"Text-analysis request uses tools."}'),
        '{"continue": true, "thought": "Select one text tool call."}',
        '{"tool_name":"text.word_count","tool_input":{"text":"modular field service workflow"}}',
    ),
    "examples/workflow/debate_pattern.py": (
        "Local models improve data control and predictable costs for many research workloads.",
        "Hosted APIs can ship faster and often provide higher quality with less ops burden.",
        (
            '{"winner":"tie","rationale":"Both positions are compelling with different tradeoffs.",'
            '"synthesis":"Use local models for sensitive data and hosted APIs for burst capacity."}'
        ),
    ),
    "examples/workflow/conversation_pattern.py": (
        ("Use a hand-crank dual-roller shelling stage with food-safe rubber rollers and a winnowing chute."),
        (
            "Add a threaded gap adjuster and quick-release side plates so farmers can tune roller "
            "spacing and clean jams quickly."
        ),
        (
            "Prototype a second concept with a peg-drum against a perforated concave, driven by "
            "gears to reduce operator force."
        ),
        (
            "Prioritize the roller prototype first because it is simpler to fabricate; validate "
            "kernel breakage, throughput, and cleaning time in field tests."
        ),
    ),
    "examples/workflow/workflow_prompt_mode.py": (
        (
            '{"title":"Deterministic workflow memo","summary":"Use one runtime that fuses core, '
            'script, and MCP tools.","priority":"high"}'
        ),
    ),
    "examples/workflow/workflow_model_step_design_tradeoff.py": (
        "Use a modular latch for faster maintenance; accept small cost increase for serviceability.",
    ),
    "examples/agents/streaming/multi_step_code_tool_calling_agent_stream.py": (
        '{"continue": true, "thought": "Run one action step."}',
        "\n".join(
            [
                'readme = call_tool("fs.read_text", {"path": "README.md", "max_bytes": 1200})',
                'stats = call_tool("text.word_count", {"text": readme["text"]})',
                'final_output = {"word_count": stats["word_count"], "summary": "README measured."}',
            ]
        ),
    ),
    "examples/agents/streaming/multi_step_json_tool_calling_agent_stream.py": (
        '{"continue": true, "thought": "Run one action step."}',
        ('{"tool_name":"text.word_count","tool_input":{"text":"README measured"},"reason":"Compute compact metric."}'),
    ),
    "examples/agents/streaming/multi_step_tool_router_agent_stream.py": (
        (
            '{"action":"TOOL_CALL","tool_names":["text.word_count"],'
            '"tool_input":{"text":"modular field service workflow"},"reason":"Compute metric."}'
        ),
        '{"action":"STOP","final_output":{"word_count":4},"reason":"done"}',
    ),
    "examples/agents/streaming/multi_step_direct_llm_agent_stream.py": (
        ('{"decision":"CONTINUE","content":"Draft answer: compute 6 * 7.","reason":"Need final wording."}'),
        '{"decision":"STOP","content":"Final answer ready.","final_output":"42","reason":"done"}',
    ),
}


if _DETERMINISTIC_MODE:

    def _resolve_example_id() -> str:
        configured_id = os.environ.get("DRA_EXAMPLE_ID")
        if isinstance(configured_id, str) and configured_id.strip():
            return configured_id.strip()

        script_path = Path(sys.argv[0])
        if script_path.is_absolute():
            try:
                repo_root = Path.cwd().resolve()
                return script_path.resolve().relative_to(repo_root).as_posix()
            except ValueError:
                return script_path.name
        return script_path.as_posix()

    class _DeterministicExampleClient:
        def __init__(self, *, response_texts: Sequence[str]) -> None:
            self._responses = list(response_texts)

        def chat(
            self,
            messages: Sequence[LLMMessage],
            *,
            model: str,
            params: LLMChatParams,
        ) -> LLMResponse:
            del messages, params
            return self._next(model)

        def stream_chat(
            self,
            messages: Sequence[LLMMessage],
            *,
            model: str,
            params: LLMChatParams,
        ) -> Iterator[LLMStreamEvent]:
            response = self.chat(messages, model=model, params=params)
            yield LLMStreamEvent(kind="delta", delta_text=response.text)
            yield LLMStreamEvent(kind="completed", response=response)

        def generate(self, request: LLMRequest) -> LLMResponse:
            return self._next(request.model or self.default_model())

        def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
            response = self.generate(request)
            yield LLMDelta(text_delta=response.text)

        def default_model(self) -> str:
            return "example-model"

        def close(self) -> None:
            return None

        def _next(self, model: str) -> LLMResponse:
            if not self._responses:
                raise RuntimeError("No deterministic responses remaining for this example script.")
            return LLMResponse(
                model=model,
                text=self._responses.pop(0),
                provider="example-test-monkeypatch",
            )

    def _patched_LlamaCppServerLLMClient() -> _DeterministicExampleClient:
        example_id = _resolve_example_id()
        responses = _SCRIPT_RESPONSES.get(example_id)
        if responses is None:
            script_name = Path(sys.argv[0]).name
            responses = _SCRIPT_RESPONSES.get(script_name)
        if responses is None:
            raise RuntimeError(f"No deterministic LLM response profile configured for '{example_id}'.")
        return _DeterministicExampleClient(response_texts=responses)

    class _DeterministicCapabilities:
        def __init__(self) -> None:
            self.streaming = False
            self.tool_calling = "best_effort"
            self.json_mode = "prompt+validate"
            self.vision = False
            self.max_context_tokens = None

    class _DeterministicBackend:
        def __init__(
            self,
            *,
            name: str,
            kind: str,
            max_retries: int,
            model_patterns: Sequence[str],
            extra_fields: dict[str, object],
        ) -> None:
            self.name = name
            self.kind = kind
            self.max_retries = max_retries
            self.model_patterns = tuple(model_patterns)
            for field_name, field_value in extra_fields.items():
                setattr(self, field_name, field_value)

        def capabilities(self) -> _DeterministicCapabilities:
            return _DeterministicCapabilities()

    class _DeterministicConfiguredClient:
        def __init__(
            self,
            *,
            name: str,
            default_model: str,
            kind: str,
            max_retries: int,
            model_patterns: Sequence[str],
            extra_backend_fields: dict[str, object],
        ) -> None:
            self._default_model = default_model
            self._backend = _DeterministicBackend(
                name=name,
                kind=kind,
                max_retries=max_retries,
                model_patterns=model_patterns,
                extra_fields=extra_backend_fields,
            )
            self._config_snapshot = {
                "name": name,
                "kind": kind,
                "default_model": default_model,
                "base_url": getattr(self._backend, "base_url", None),
                "max_retries": max_retries,
                "model_patterns": list(model_patterns),
            }
            for field_name, field_value in extra_backend_fields.items():
                normalized_name = field_name[1:] if field_name.startswith("_") else field_name
                self._config_snapshot[normalized_name] = field_value

        def default_model(self) -> str:
            return self._default_model

        def capabilities(self) -> _DeterministicCapabilities:
            return self._backend.capabilities()

        def config_snapshot(self) -> dict[str, object]:
            return dict(self._config_snapshot)

        def server_snapshot(self) -> None:
            return None

        def describe(self) -> dict[str, object]:
            capabilities = self.capabilities()
            return {
                "client_class": self.__class__.__name__,
                "default_model": self.default_model(),
                "backend": self.config_snapshot(),
                "capabilities": {
                    "streaming": capabilities.streaming,
                    "tool_calling": capabilities.tool_calling,
                    "json_mode": capabilities.json_mode,
                    "vision": capabilities.vision,
                    "max_context_tokens": capabilities.max_context_tokens,
                },
                "server": self.server_snapshot(),
            }

        def close(self) -> None:
            return None

    def _patched_MlxLocalLLMClient(**kwargs: object) -> _DeterministicConfiguredClient:
        default_model = kwargs.get("default_model", kwargs.get("model_id", "example-model"))
        if not isinstance(default_model, str):
            default_model = "example-model"
        name = str(kwargs.get("name", "mlx-local"))
        raw_model_patterns = kwargs.get("model_patterns", ())
        model_patterns = (
            tuple(raw_model_patterns)
            if isinstance(raw_model_patterns, Sequence) and not isinstance(raw_model_patterns, str)
            else ()
        )
        max_retries_raw = kwargs.get("max_retries", 2)
        max_retries = int(max_retries_raw) if isinstance(max_retries_raw, int) else 2
        return _DeterministicConfiguredClient(
            name=name,
            default_model=default_model,
            kind="mlx_local",
            max_retries=max_retries,
            model_patterns=model_patterns,
            extra_backend_fields={
                "_model_id": kwargs.get("model_id", default_model),
                "_quantization": kwargs.get("quantization", "4bit"),
            },
        )

    llm_module.LlamaCppServerLLMClient = _patched_LlamaCppServerLLMClient
    llm_module.MlxLocalLLMClient = _patched_MlxLocalLLMClient

    # Patch top-level exported accessor for tests that import it directly.
    try:
        import design_research_agents as package_api
    except Exception:
        package_api = None
    if package_api is not None:
        package_api.LlamaCppServerLLMClient = _patched_LlamaCppServerLLMClient
        package_api.MlxLocalLLMClient = _patched_MlxLocalLLMClient
