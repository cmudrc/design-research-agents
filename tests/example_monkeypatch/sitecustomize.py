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
    "examples/agents/direct_llm_with_pinned_skills.py": (
        "Repairability priorities: use accessible fasteners, clear service order, and fast battery swaps.",
    ),
    "examples/agents/multi_step_code_tool_calling_agent.py": (
        "\n".join(
            [
                'stats = call_tool("text.word_count", {"text": "design review metrics"})',
                'final_answer({"word_count": stats["word_count"]})',
            ]
        ),
    ),
    "examples/agents/multi_step_json_tool_calling_agent.py": (
        (
            '{"tool_name":"text.word_count","tool_input":{"text":"design research agents"},'
            '"reason":"Measure the phrase before answering."}'
        ),
        ('{"tool_name":"final_answer","tool_input":{"word_count":3},"reason":"done"}'),
    ),
    "examples/workflow/demo_json_tool_workflow.py": (
        (
            '{"tool_name":"text.word_count","tool_input":{"text":"design research workshop"},'
            '"reason":"Measure the phrase before answering."}'
        ),
        ('{"tool_name":"final_answer","tool_input":{"word_count":3},"reason":"done"}'),
    ),
    "examples/agents/multi_step_json_with_skills.py": (
        (
            '{"tool_name":"skills.activate","tool_input":{"skill_name":"word_count_helper"},'
            '"reason":"Load the local counting instructions first."}'
        ),
        (
            '{"tool_name":"text.word_count","tool_input":{"text":"design research agents"},'
            '"reason":"Measure the phrase before answering."}'
        ),
        ('{"tool_name":"final_answer","tool_input":{"word_count":3},"reason":"done"}'),
    ),
    "examples/agents/multi_step_json_with_memory.py": (
        (
            '{"tool_name":"text.word_count","tool_input":{"text":"Prior design note: target quick '
            'maintenance by minimizing tool changes and favoring reusable fasteners."}}'
        ),
        '{"tool_name":"final_answer","tool_input":{"word_count":14},"reason":"done"}',
    ),
    "examples/agents/multi_step_direct_llm_agent.py": (
        '{"decision":"CONTINUE","content":"Draft answer: compute 6 * 7.","reason":"Need final wording."}',
        '{"decision":"STOP","content":"Final answer ready.","final_output":"42","reason":"done"}',
    ),
    "examples/optimization/multi_step_json_tool_calling_1d_optimization.py": (
        '{"tool_name":"optimizer.evaluate","tool_input":{"x":3},"reason":"Start from the given point."}',
        '{"tool_name":"optimizer.evaluate","tool_input":{"x":1},"reason":"Check a better point closer to zero."}',
        '{"tool_name":"optimizer.evaluate","tool_input":{"x":0},"reason":"Test the obvious minimum."}',
        '{"tool_name":"final_answer","tool_input":{"best_x":0,"best_objective":0,"evaluations":3},"reason":"done"}',
    ),
    "examples/patterns/plan_execute.py": (
        (
            '{"steps":[{"step_id":"count_phrase_words","instruction":"Call text.word_count on '
            '\'design system research workflow\' and return only word_count.","success_criteria":'
            '"Return the exact word count."}]}'
        ),
        (
            '{"tool_name":"text.word_count","tool_input":{"text":"design system research '
            'workflow"},"reason":"Count the phrase before answering."}'
        ),
        '{"tool_name":"final_answer","tool_input":{"word_count":4},"reason":"done"}',
    ),
    "examples/patterns/propose_critic.py": (
        "Draft v1: simple proposal.",
        '{"approved": false, "feedback": "Add more detail.", "revision_goals": ["expand rationale"]}',
        "Draft v2: proposal with more detail.",
        '{"approved": true, "feedback": "Looks good.", "revision_goals": []}',
    ),
    "examples/patterns/router_delegate.py": (
        '{"tool_name":"json_tool_agent","tool_input":{},"reason":"Text-analysis request uses tools."}',
        '{"tool_name":"text.word_count","tool_input":{"text":"modular field service workflow"}}',
        '{"tool_name":"final_answer","tool_input":{"word_count":4},"reason":"done"}',
    ),
    "examples/patterns/debate_pattern.py": (
        "Local models improve data control and predictable costs for many research workloads.",
        "Hosted APIs can ship faster and often provide higher quality with less ops burden.",
        (
            '{"winner":"tie","rationale":"Both positions are compelling with different tradeoffs.",'
            '"synthesis":"Use local models for sensitive data and hosted APIs for burst capacity."}'
        ),
    ),
    "examples/patterns/two_speaker_conversation.py": (
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
    "examples/patterns/coordination_patterns.py": (
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
    "examples/patterns/tree_search.py": (
        (
            '{"candidates":[{"concept":"lightweight frame","tradeoff":"low mass"},'
            '{"concept":"modular frame","tradeoff":"serviceability"}]}'
        ),
        '{"score":0.44}',
        '{"score":0.73}',
        (
            '{"candidates":[{"concept":"modular frame + keyed hatch","tradeoff":"maintainability"},'
            '{"concept":"modular frame + fail-safe latch","tradeoff":"reliability"}]}'
        ),
        '{"score":0.79}',
        '{"score":0.91}',
    ),
    "examples/patterns/ralph_loop.py": (
        '{"proposal":"Draft v1 with modular panels","changes":["add service hatch"]}',
        '{"risks":["hatch alignment unclear"],"advice":["add keyed guides"]}',
        '{"synthesis":"Draft v1 + keyed guides","actions":["spec hatch alignment"]}',
        '{"score":0.62,"rationale":"initial draft still missing service procedure details"}',
        ('{"proposal":"Draft v2 with keyed guides and service sequence","changes":["add torque notes"]}'),
        '{"risks":["tooling variation"],"advice":["standardize fastener heads"]}',
        ('{"synthesis":"Draft v2 standardized fasteners and service sequence","actions":["freeze fastener standard"]}'),
        '{"score":0.88,"rationale":"consensus threshold met"}',
    ),
    "examples/patterns/nominal_team.py": (
        ('{"concept":"split enclosure with captive screws","strengths":["fast field access"],"risks":["gasket wear"]}'),
        (
            '{"concept":"sealed cartridge bay with keyed latch","strengths":["weather sealing"],'
            '"risks":["latch tolerance stack"]}'
        ),
        ('{"concept":"sheet-metal drawer module","strengths":["simple fabrication"],"risks":["larger envelope size"]}'),
        (
            '{"best_member_id":"repairability","scores":{"repairability":0.91,"reliability":0.74,'
            '"manufacturability":0.68},"rationale":"Best maintenance turnaround under the active task."}'
        ),
    ),
    "examples/patterns/rag.py": (
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
    "examples/clients/demo_client.py": ("Run a quick affinity-mapping exercise with a shared set of interview notes.",),
    "examples/clients/anthropic_service_client.py": (
        "Run architecture red-team reviews before committing high-impact changes with uncertain failure modes.",
    ),
    "examples/clients/gemini_service_client.py": (
        "Run a design pre-mortem before committing architecture changes with high uncertainty or safety risk.",
    ),
    "examples/clients/groq_service_client.py": (
        "Prefer deeper review when architectural choices are expensive to reverse.",
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
    "examples/workflow/agent_routing.py": "examples/patterns/router_delegate.py",
    "examples/workflow/debate_pattern.py": "examples/patterns/debate_pattern.py",
    "examples/workflow/conversation_pattern.py": "examples/patterns/two_speaker_conversation.py",
    "examples/workflow/networked_blackboard.py": "examples/patterns/coordination_patterns.py",
    "examples/workflow/tree_search.py": "examples/patterns/tree_search.py",
    "examples/workflow/rag.py": "examples/patterns/rag.py",
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

        def __enter__(self) -> _DeterministicExampleClient:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            self.close()
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

    def _patched_demo_client(**kwargs: object) -> _DeterministicConfiguredClient:
        model = _as_name(kwargs.get("api_model"), "qwen3-0.6b-q8-demo")
        return _DeterministicConfiguredClient(
            client_class_name="DemoLLMClient",
            name=_as_name(kwargs.get("name"), "demo-local"),
            default_model=model,
            kind="llama_cpp_server",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "host": _as_name(kwargs.get("host"), "127.0.0.1"),
                "port": kwargs.get("port", 8001),
                "api_model": model,
                "thinking": _as_name(kwargs.get("thinking"), "off"),
            },
            server_snapshot={
                "managed": True,
                "kind": "llama_cpp_server",
                "host": _as_name(kwargs.get("host"), "127.0.0.1"),
                "port": kwargs.get("port", 8001),
            },
        )

    def _patched_mlx_local_client(**kwargs: object) -> _DeterministicConfiguredClient:
        default_model = _as_name(kwargs.get("default_model"), _as_name(kwargs.get("model_id"), "example-model"))
        return _DeterministicConfiguredClient(
            client_class_name="MLXLocalLLMClient",
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

    def _patched_anthropic_service_client(**kwargs: object) -> _DeterministicConfiguredClient:
        default_model = _as_name(kwargs.get("default_model"), "claude-3-5-haiku-latest")
        return _DeterministicConfiguredClient(
            client_class_name="AnthropicServiceLLMClient",
            name=_as_name(kwargs.get("name"), "anthropic-service"),
            default_model=default_model,
            kind="anthropic_service",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "base_url": _as_name(kwargs.get("base_url"), "https://api.anthropic.com"),
                "api_key_env": _as_name(kwargs.get("api_key_env"), "ANTHROPIC_API_KEY"),
            },
            server_snapshot=None,
        )

    def _patched_gemini_service_client(**kwargs: object) -> _DeterministicConfiguredClient:
        default_model = _as_name(kwargs.get("default_model"), "gemini-2.5-flash")
        return _DeterministicConfiguredClient(
            client_class_name="GeminiServiceLLMClient",
            name=_as_name(kwargs.get("name"), "gemini-service"),
            default_model=default_model,
            kind="gemini_service",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "api_key_env": _as_name(kwargs.get("api_key_env"), "GOOGLE_API_KEY"),
            },
            server_snapshot=None,
        )

    def _patched_groq_service_client(**kwargs: object) -> _DeterministicConfiguredClient:
        default_model = _as_name(kwargs.get("default_model"), "llama-3.1-8b-instant")
        return _DeterministicConfiguredClient(
            client_class_name="GroqServiceLLMClient",
            name=_as_name(kwargs.get("name"), "groq-service"),
            default_model=default_model,
            kind="groq_service",
            max_retries=_as_int(kwargs.get("max_retries"), 2),
            model_patterns=_as_patterns(kwargs.get("model_patterns")),
            extra_backend_fields={
                "base_url": _as_name(kwargs.get("base_url"), "https://api.groq.com"),
                "api_key_env": _as_name(kwargs.get("api_key_env"), "GROQ_API_KEY"),
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
            client_class_name="VLLMServerLLMClient",
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
            client_class_name="SGLangServerLLMClient",
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
        "DemoLLMClient": _patched_demo_client,
        "LlamaCppServerLLMClient": _patched_llama_cpp_server_client,
        "MLXLocalLLMClient": _patched_mlx_local_client,
        "OllamaLLMClient": _patched_ollama_client,
        "OpenAICompatibleHTTPLLMClient": _patched_openai_compatible_http_client,
        "OpenAIServiceLLMClient": _patched_openai_service_client,
        "AnthropicServiceLLMClient": _patched_anthropic_service_client,
        "GeminiServiceLLMClient": _patched_gemini_service_client,
        "GroqServiceLLMClient": _patched_groq_service_client,
        "TransformersLocalLLMClient": _patched_transformers_local_client,
        "VLLMServerLLMClient": _patched_vllm_server_client,
        "SGLangServerLLMClient": _patched_sglang_server_client,
    }

    for class_name, factory in _PATCHES.items():
        setattr(llm_module, class_name, factory)
        setattr(llm_clients_module, class_name, factory)
        setattr(package_api, class_name, factory)
