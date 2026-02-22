"""Test-only monkeypatches for running examples with deterministic LLM responses.

Loaded automatically by Python when this directory is prepended to ``PYTHONPATH``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

_DETERMINISTIC_MODE = os.environ.get("DRA_EXAMPLE_LLM_MODE", "").strip().lower() == "deterministic"

if _DETERMINISTIC_MODE:
    import design_research_agents as package_api
    import design_research_agents.llm as llm_module
    import design_research_agents.llm.clients as llm_clients_module
    from design_research_agents._contracts import LLMChatParams, LLMDelta, LLMStreamEvent
    from design_research_agents.llm import LLMMessage, LLMRequest, LLMResponse


_SCRIPT_RESPONSE_PROFILES: dict[str, tuple[str, ...]] = {
    "examples/agents/direct_llm_call.py": ("4",),
    "examples/agents/multi_step_code_tool_calling_agent.py": (
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
    "examples/agents/multi_step_json_tool_calling_agent.py": (
        '{"continue": true, "thought": "Read README first."}',
        (
            '{"tool_name":"fs.read_text","tool_input":{"path":"README.md","max_bytes":800},'
            '"reason":"Need repository text context."}'
        ),
        '{"continue": false, "thought": "Task complete."}',
    ),
    "examples/agents/multi_step_json_with_memory.py": (
        '{"continue": true, "thought": "start"}',
        '{"tool_name": "text.word_count", "tool_input": {"text": "design context memory"}}',
        '{"continue": false, "thought": "done"}',
    ),
    "examples/agents/multi_step_direct_llm_agent.py": (
        '{"decision":"CONTINUE","content":"Draft answer: compute 6 * 7.","reason":"Need final wording."}',
        '{"decision":"STOP","content":"Final answer ready.","final_output":"42","reason":"done"}',
    ),
    "examples/optimization/multi_step_tool_router_1d_optimization.py": (
        '{"continue": true, "thought": "Start descending toward zero."}',
        '{"tool_name":"optimizer.decrease_x","tool_input":{"step":1},"reason":"Decrease x toward zero."}',
        '{"continue": true, "thought": "Still above zero, continue decreasing."}',
        '{"tool_name":"optimizer.decrease_x","tool_input":{"step":1},"reason":"Keep moving toward zero."}',
        '{"continue": true, "thought": "One more decrease should reach zero."}',
        '{"tool_name":"optimizer.decrease_x","tool_input":{"step":1},"reason":"Reach x=0."}',
        '{"continue": false, "thought": "No better one-step move remains."}',
    ),
    "examples/patterns/plan_execute.py": (
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
    "examples/patterns/propose_critic.py": (
        "Draft v1: simple proposal.",
        '{"approved": false, "feedback": "Add more detail.", "revision_goals": ["expand rationale"]}',
        "Draft v2: proposal with more detail.",
        '{"approved": true, "feedback": "Looks good.", "revision_goals": []}',
    ),
    "examples/patterns/agent_routing.py": (
        '{"action":"TOOL_CALL","tool_names":["json_tool_agent"],"reason":"Text-analysis request uses tools."}',
        '{"continue": true, "thought": "Select one text tool call."}',
        '{"tool_name":"text.word_count","tool_input":{"text":"modular field service workflow"}}',
    ),
    "examples/patterns/debate_pattern.py": (
        "Local models improve data control and predictable costs for many research workloads.",
        "Hosted APIs can ship faster and often provide higher quality with less ops burden.",
        (
            '{"winner":"tie","rationale":"Both positions are compelling with different tradeoffs.",'
            '"synthesis":"Use local models for sensitive data and hosted APIs for burst capacity."}'
        ),
    ),
    "examples/patterns/conversation_pattern.py": (
        "Use a hand-crank dual-roller shelling stage with food-safe rubber rollers and a winnowing chute.",
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
    "examples/patterns/networked_blackboard.py": (
        "Peer contribution: prioritize captive screws for quicker service loops.",
        "Peer contribution: keep gasket alignment features for resealing reliability.",
        "Peer contribution: propose tool-less battery tray removal path.",
        "Peer contribution: add visual fastener indexing for field technicians.",
        "Peer contribution: standardize fastener head geometry across modules.",
        "Peer contribution: preserve ingress protection while reducing disassembly steps.",
        "Peer contribution: compare option A and option B maintenance sequences.",
        "Peer contribution: select the concept with fastest validated service sequence.",
        "Peer contribution: document final maintenance SOP candidates.",
        "Peer contribution: finalize blackboard recommendation summary.",
    ),
    "examples/patterns/rag_reasoning.py": (
        "Prioritize maintainability checks and explicit testability criteria in the recommended architecture.",
    ),
    "examples/workflow/workflow_prompt_mode.py": (
        (
            '{"title":"Deterministic workflow memo","summary":"Use one runtime that '
            'fuses core, script, and MCP tools.","priority":"high"}'
        ),
    ),
    "examples/workflow/workflow_model_step_design_tradeoff.py": (
        "Use a modular latch for faster maintenance; accept small cost increase for serviceability.",
    ),
    "examples/workflow/workflow_delegate_and_memory_steps.py": (
        "Use captive screws with standardized head type for faster maintenance.",
        "Add gasket alignment features to preserve ingress protection after service.",
    ),
    "examples/clients/llama_cpp_server_client.py": (
        "Tradeoff: strict review gates improve reliability but can slow delivery speed.",
    ),
    "examples/clients/mlx_local_client.py": (
        "Keep schema fields stable, documented, and versioned for comparability.",
    ),
    "examples/clients/openai_compatible_http_client.py": (
        "Use fast drafts for iteration, then escalate critical decisions to higher-quality models.",
    ),
    "examples/clients/openai_service_client.py": (
        "Use multi-agent critique when decisions have high risk and need diverse failure analysis.",
    ),
    "examples/clients/transformers_local_client.py": (
        "Deterministic local runs make design comparisons repeatable across experiments.",
    ),
    "examples/clients/vllm_server_client.py": (
        "Local serving reduces backend drift and improves benchmark reproducibility.",
    ),
    "examples/clients/ollama_local_client.py": (
        "Use automated local pulls when startup reliability matters more than cold-start time.",
    ),
    "examples/clients/sglang_server_client.py": (
        "SGLang-style serving helps when you need stable local throughput for repeated tests.",
    ),
}

_PATH_ALIASES: dict[str, str] = {
    "examples/agents/basic/direct_llm_call.py": "examples/agents/direct_llm_call.py",
    "examples/agents/basic/multi_step_code_tool_calling_agent.py": (
        "examples/agents/multi_step_code_tool_calling_agent.py"
    ),
    "examples/agents/basic/multi_step_json_tool_calling_agent.py": (
        "examples/agents/multi_step_json_tool_calling_agent.py"
    ),
    "examples/agents/basic/multi_step_json_with_memory.py": "examples/agents/multi_step_json_with_memory.py",
    "examples/agents/basic/multi_step_direct_llm_agent.py": "examples/agents/multi_step_direct_llm_agent.py",
    "examples/workflow/plan_execute.py": "examples/patterns/plan_execute.py",
    "examples/workflow/propose_critic.py": "examples/patterns/propose_critic.py",
    "examples/workflow/agent_routing.py": "examples/patterns/agent_routing.py",
    "examples/workflow/debate_pattern.py": "examples/patterns/debate_pattern.py",
    "examples/workflow/conversation_pattern.py": "examples/patterns/conversation_pattern.py",
    "examples/workflow/networked_blackboard.py": "examples/patterns/networked_blackboard.py",
    "examples/workflow/tree_search.py": "examples/patterns/tree_search.py",
    "examples/workflow/rag_reasoning.py": "examples/patterns/rag_reasoning.py",
}


def _normalize_example_key(raw_value: str) -> str:
    normalized = raw_value.strip().replace("\\", "/")
    if not normalized:
        return ""

    path = Path(normalized)
    if path.is_absolute():
        try:
            normalized = path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            normalized = path.as_posix()

    if normalized.startswith("./"):
        normalized = normalized[2:]

    marker = "examples/"
    marker_index = normalized.find(marker)
    if marker_index >= 0:
        normalized = normalized[marker_index:]

    return normalized


def _resolve_example_id() -> str:
    configured_id = os.environ.get("DRA_EXAMPLE_ID")
    if isinstance(configured_id, str) and configured_id.strip():
        return _normalize_example_key(configured_id)
    return _normalize_example_key(sys.argv[0])


def _resolve_profile(example_id: str) -> tuple[str, ...]:
    direct_profile = _SCRIPT_RESPONSE_PROFILES.get(example_id)
    if direct_profile is not None:
        return direct_profile

    aliased_id = _PATH_ALIASES.get(example_id)
    if aliased_id is not None:
        aliased_profile = _SCRIPT_RESPONSE_PROFILES.get(aliased_id)
        if aliased_profile is not None:
            return aliased_profile

    basename = Path(example_id).name
    if basename:
        for key, profile in _SCRIPT_RESPONSE_PROFILES.items():
            if Path(key).name == basename:
                return profile

    return ("deterministic example response",)


if _DETERMINISTIC_MODE:

    class _DeterministicExampleClient:
        def __init__(self, *, response_texts: Sequence[str], default_model: str) -> None:
            self._responses = list(response_texts)
            self._default_model = default_model

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
            model = request.model or self.default_model()
            return self._next(model)

        def stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
            response = self.generate(request)
            yield LLMDelta(text_delta=response.text)

        def default_model(self) -> str:
            return self._default_model

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

    class _DeterministicConfiguredClient(_DeterministicExampleClient):
        def __init__(
            self,
            *,
            client_class_name: str,
            name: str,
            default_model: str,
            kind: str,
            max_retries: int,
            model_patterns: Sequence[str],
            extra_backend_fields: dict[str, object],
            server_snapshot: dict[str, object] | None,
        ) -> None:
            profile = _resolve_profile(_resolve_example_id())
            super().__init__(response_texts=profile, default_model=default_model)
            self._backend = _DeterministicBackend(
                name=name,
                kind=kind,
                max_retries=max_retries,
                model_patterns=model_patterns,
                extra_fields=extra_backend_fields,
            )
            self._client_class_name = client_class_name
            self._server_snapshot = dict(server_snapshot) if isinstance(server_snapshot, dict) else None
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

        def capabilities(self) -> _DeterministicCapabilities:
            return self._backend.capabilities()

        def config_snapshot(self) -> dict[str, object]:
            return dict(self._config_snapshot)

        def server_snapshot(self) -> dict[str, object] | None:
            return dict(self._server_snapshot) if self._server_snapshot is not None else None

        def describe(self) -> dict[str, object]:
            capabilities = self.capabilities()
            return {
                "client_class": self._client_class_name,
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

    def _as_name(value: object, fallback: str) -> str:
        return value if isinstance(value, str) and value.strip() else fallback

    def _as_int(value: object, fallback: int) -> int:
        return value if isinstance(value, int) else fallback

    def _as_patterns(value: object) -> tuple[str, ...]:
        if isinstance(value, Sequence) and not isinstance(value, str):
            return tuple(str(item) for item in value)
        return ()

    def _patched_llama_cpp_server_client(**kwargs: object) -> _DeterministicConfiguredClient:
        model = _as_name(kwargs.get("api_model"), _as_name(kwargs.get("model"), "example-model"))
        return _DeterministicConfiguredClient(
            client_class_name="LlamaCppServerLLMClient",
            name=_as_name(kwargs.get("name"), "llama-cpp"),
            default_model=model,
            kind="llama_cpp_server",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "host": _as_name(kwargs.get("host"), "127.0.0.1"),
                "port": kwargs.get("port", 8011),
                "api_model": model,
            },
            server_snapshot={
                "managed": True,
                "kind": "llama_cpp_server",
                "host": _as_name(kwargs.get("host"), "127.0.0.1"),
                "port": kwargs.get("port", 8011),
            },
        )

    def _patched_mlx_local_client(**kwargs: object) -> _DeterministicConfiguredClient:
        default_model = _as_name(kwargs.get("default_model"), _as_name(kwargs.get("model_id"), "example-model"))
        return _DeterministicConfiguredClient(
            client_class_name="MlxLocalLLMClient",
            name=_as_name(kwargs.get("name"), "mlx-local"),
            default_model=default_model,
            kind="mlx_local",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "model_id": _as_name(kwargs.get("model_id"), default_model),
                "quantization": _as_name(kwargs.get("quantization"), "4bit"),
            },
            server_snapshot=None,
        )

    def _patched_ollama_client(**kwargs: object) -> _DeterministicConfiguredClient:
        default_model = _as_name(kwargs.get("default_model"), "example-model")
        host = _as_name(kwargs.get("host"), "127.0.0.1")
        port = kwargs.get("port", 11434)
        return _DeterministicConfiguredClient(
            client_class_name="OllamaLLMClient",
            name=_as_name(kwargs.get("name"), "ollama"),
            default_model=default_model,
            kind="ollama",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "base_url": f"http://{host}:{port}",
                "host": host,
                "port": port,
            },
            server_snapshot={
                "managed": bool(kwargs.get("manage_server", True)),
                "kind": "ollama",
                "host": host,
                "port": port,
            },
        )

    def _patched_openai_compatible_http_client(**kwargs: object) -> _DeterministicConfiguredClient:
        default_model = _as_name(kwargs.get("default_model"), "example-model")
        return _DeterministicConfiguredClient(
            client_class_name="OpenAICompatibleHTTPLLMClient",
            name=_as_name(kwargs.get("name"), "openai-compatible-http"),
            default_model=default_model,
            kind="openai_compatible_http",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "base_url": _as_name(kwargs.get("base_url"), "http://127.0.0.1:8000/v1"),
                "api_key_env": _as_name(kwargs.get("api_key_env"), "OPENAI_API_KEY"),
            },
            server_snapshot=None,
        )

    def _patched_openai_service_client(**kwargs: object) -> _DeterministicConfiguredClient:
        default_model = _as_name(kwargs.get("default_model"), "gpt-4o-mini")
        return _DeterministicConfiguredClient(
            client_class_name="OpenAIServiceLLMClient",
            name=_as_name(kwargs.get("name"), "openai-service"),
            default_model=default_model,
            kind="openai_service",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "base_url": _as_name(kwargs.get("base_url"), "https://api.openai.com/v1"),
                "api_key_env": _as_name(kwargs.get("api_key_env"), "OPENAI_API_KEY"),
            },
            server_snapshot=None,
        )

    def _patched_transformers_local_client(**kwargs: object) -> _DeterministicConfiguredClient:
        default_model = _as_name(kwargs.get("default_model"), _as_name(kwargs.get("model_id"), "example-model"))
        return _DeterministicConfiguredClient(
            client_class_name="TransformersLocalLLMClient",
            name=_as_name(kwargs.get("name"), "transformers-local"),
            default_model=default_model,
            kind="transformers_local",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "model_id": _as_name(kwargs.get("model_id"), default_model),
                "device": _as_name(kwargs.get("device"), "auto"),
                "dtype": _as_name(kwargs.get("dtype"), "auto"),
                "quantization": _as_name(kwargs.get("quantization"), "none"),
            },
            server_snapshot=None,
        )

    def _patched_vllm_server_client(**kwargs: object) -> _DeterministicConfiguredClient:
        api_model = _as_name(kwargs.get("api_model"), _as_name(kwargs.get("model"), "example-model"))
        host = _as_name(kwargs.get("host"), "127.0.0.1")
        port = kwargs.get("port", 8002)
        return _DeterministicConfiguredClient(
            client_class_name="VllmServerLLMClient",
            name=_as_name(kwargs.get("name"), "vllm"),
            default_model=api_model,
            kind="vllm_server",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "base_url": f"http://{host}:{port}/v1",
                "host": host,
                "port": port,
            },
            server_snapshot={
                "managed": bool(kwargs.get("manage_server", True)),
                "kind": "vllm_server",
                "host": host,
                "port": port,
            },
        )

    def _patched_sglang_server_client(**kwargs: object) -> _DeterministicConfiguredClient:
        model = _as_name(kwargs.get("model"), "example-model")
        host = _as_name(kwargs.get("host"), "127.0.0.1")
        port = kwargs.get("port", 30000)
        return _DeterministicConfiguredClient(
            client_class_name="SglangServerLLMClient",
            name=_as_name(kwargs.get("name"), "sglang"),
            default_model=model,
            kind="sglang_server",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "base_url": f"http://{host}:{port}/v1",
                "host": host,
                "port": port,
            },
            server_snapshot={
                "managed": bool(kwargs.get("manage_server", True)),
                "kind": "sglang_server",
                "host": host,
                "port": port,
            },
        )

    _PATCHES = {
        "LlamaCppServerLLMClient": _patched_llama_cpp_server_client,
        "MlxLocalLLMClient": _patched_mlx_local_client,
        "OllamaLLMClient": _patched_ollama_client,
        "OpenAICompatibleHTTPLLMClient": _patched_openai_compatible_http_client,
        "OpenAIServiceLLMClient": _patched_openai_service_client,
        "TransformersLocalLLMClient": _patched_transformers_local_client,
        "VllmServerLLMClient": _patched_vllm_server_client,
        "SglangServerLLMClient": _patched_sglang_server_client,
    }

    for class_name, factory in _PATCHES.items():
        setattr(llm_module, class_name, factory)
        setattr(llm_clients_module, class_name, factory)
        setattr(package_api, class_name, factory)
