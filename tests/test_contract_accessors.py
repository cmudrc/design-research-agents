"""Contract tests for read-only payload helpers and LLM client lifecycle APIs."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence

import pytest

import design_research_agents.llm as dra_llm
from design_research_agents._contracts import (
    BackendCapabilities,
    BackendStatus,
    LLMClient,
    LLMDelta,
    LLMRequest,
    LLMResponse,
    MemoryRecord,
    MemorySearchQuery,
    MemoryStore,
    MemoryWriteRecord,
    ToolArtifact,
    ToolError,
    ToolResult,
    WorkflowStepResult,
)
from design_research_agents.llm import (
    AnthropicServiceLLMClient,
    AzureOpenAIServiceLLMClient,
    GeminiServiceLLMClient,
    GroqServiceLLMClient,
    HTMLLLMClient,
    OpenAICompatibleHTTPLLMClient,
    OpenAIServiceLLMClient,
)
from design_research_agents.llm._backends._base import BaseLLMBackend
from design_research_agents.llm.clients._shared import _SingleBackendLLMClient


class _StubBackend(BaseLLMBackend):
    def __init__(self) -> None:
        super().__init__(
            name="stub",
            kind="test",
            default_model="stub-model",
            base_url=None,
            config_hash="stub-hash",
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            streaming=False,
            tool_calling="none",
            json_mode="none",
            vision=False,
            max_context_tokens=None,
        )

    def healthcheck(self) -> BackendStatus:
        return BackendStatus(ok=True)

    def _generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            text="ok",
            model=request.model,
            provider="stub",
        )

    def _stream(self, request: LLMRequest) -> Iterator[LLMDelta]:
        del request
        if False:
            yield LLMDelta(text_delta=None)


class _StubMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self.closed = False

    def write(
        self,
        records: Sequence[MemoryWriteRecord],
        *,
        namespace: str = "default",
    ) -> list[MemoryRecord]:
        del records, namespace
        return []

    def search(self, query: MemorySearchQuery) -> list[MemoryRecord]:
        del query
        return []

    def close(self) -> None:
        self.closed = True
        super().close()


def test_workflow_step_result_output_accessors_mirror_execution_result_helpers() -> None:
    result = WorkflowStepResult(
        step_id="step-1",
        status="completed",
        success=True,
        output={
            "mapping_value": {"nested": "value"},
            "list_value": ["a", "b"],
            "tuple_value": ("x", "y"),
            "scalar_value": 7,
        },
    )

    assert result.output_value("scalar_value") == 7
    assert result.output_value("missing", "fallback") == "fallback"

    mapping_value = result.output_dict("mapping_value")
    assert mapping_value == {"nested": "value"}
    mapping_value["nested"] = "changed"
    assert result.output["mapping_value"] == {"nested": "value"}
    assert result.output_dict("scalar_value") == {}

    list_value = result.output_list("list_value")
    assert list_value == ["a", "b"]
    list_value.append("c")
    assert result.output["list_value"] == ["a", "b"]
    assert result.output_list("tuple_value") == ["x", "y"]
    assert result.output_list("scalar_value") == []


def test_tool_result_read_accessors_normalize_common_payload_shapes() -> None:
    result = ToolResult(
        tool_name="demo",
        ok=True,
        result={"payload": "ok"},
        artifacts=(
            ToolArtifact(path="artifacts/one.txt", mime="text/plain"),
            {"path": "artifacts/two.json", "mime": "application/json"},
        ),
    )

    result_dict = result.result_dict()
    assert result_dict == {"payload": "ok"}
    result_dict["payload"] = "changed"
    assert result.result == {"payload": "ok"}
    assert result.result_list() == []
    assert result.error_message is None
    assert result.artifact_paths == ("artifacts/one.txt", "artifacts/two.json")


@pytest.mark.parametrize(
    ("payload", "expected_list"),
    [
        (["a", "b"], ["a", "b"]),
        (("x", "y"), ["x", "y"]),
        ("not-a-list", []),
    ],
)
def test_tool_result_result_list_handles_list_tuple_and_scalar(payload: object, expected_list: list[object]) -> None:
    result = ToolResult(tool_name="demo", ok=True, result=payload)
    normalized_list = result.result_list()
    assert normalized_list == expected_list
    if isinstance(payload, list):
        normalized_list.append("extra")
        assert result.result == ["a", "b"]
    assert result.result_dict() == {}


@pytest.mark.parametrize(
    ("error_input", "expected_message"),
    [
        (ToolError(type="Explicit", message="prebuilt"), "prebuilt"),
        ({"type": "Mapped", "message": "mapping"}, "mapping"),
        ("string-error", "string-error"),
        (None, None),
    ],
)
def test_tool_result_error_message_reflects_normalized_error_payload(
    error_input: ToolError | dict[str, object] | str | None,
    expected_message: str | None,
) -> None:
    result = ToolResult(tool_name="demo", ok=False, error=error_input)
    assert result.error_message == expected_message


@pytest.mark.parametrize(
    "client_name",
    [name for name in dra_llm.__all__ if name.endswith("LLMClient")],
)
def test_all_public_llm_client_classes_expose_introspection_methods(client_name: str) -> None:
    client_cls = getattr(dra_llm, client_name)

    for method_name in (
        "close",
        "__enter__",
        "__exit__",
        "capabilities",
        "config_snapshot",
        "server_snapshot",
        "describe",
    ):
        assert callable(getattr(client_cls, method_name, None))


def test_single_backend_llm_client_supports_context_manager_lifecycle() -> None:
    client = _SingleBackendLLMClient(backend=_StubBackend())

    with client as opened_client:
        assert opened_client is client
        assert opened_client.default_model() == "stub-model"


def test_memory_store_protocol_supplies_default_context_manager_lifecycle() -> None:
    store = _StubMemoryStore()

    assert store.__enter__() is store
    assert store.__exit__(None, None, None) is None
    assert store.closed is True


@pytest.mark.parametrize(
    "client_factory",
    [
        OpenAIServiceLLMClient,
        AzureOpenAIServiceLLMClient,
        AnthropicServiceLLMClient,
        GeminiServiceLLMClient,
        GroqServiceLLMClient,
        HTMLLLMClient,
        OpenAICompatibleHTTPLLMClient,
    ],
    ids=[
        "openai",
        "azure-openai",
        "anthropic",
        "gemini",
        "groq",
        "html",
        "openai-compatible",
    ],
)
def test_representative_public_llm_clients_return_introspection_payloads(
    client_factory: Callable[[], LLMClient],
) -> None:
    with client_factory() as client:
        capabilities = client.capabilities()
        assert isinstance(capabilities, BackendCapabilities)

        config_snapshot = client.config_snapshot()
        assert isinstance(config_snapshot, Mapping)
        assert config_snapshot

        server_snapshot = client.server_snapshot()
        assert server_snapshot is None or isinstance(server_snapshot, Mapping)

        description = client.describe()
        assert isinstance(description, Mapping)
        assert description["client_class"] == client.__class__.__name__
        assert description["default_model"] == client.default_model()
        assert isinstance(description["backend"], Mapping)
        assert isinstance(description["capabilities"], Mapping)
