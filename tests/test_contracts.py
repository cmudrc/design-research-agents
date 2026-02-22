from __future__ import annotations

import subprocess
from collections.abc import Sequence
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from design_research_agents._contracts._execution import ExecutionResult
from design_research_agents._contracts._memory import (
    MemoryRecord,
    MemorySearchQuery,
    MemoryStore,
    MemoryWriteRecord,
)
from design_research_agents._contracts._workflow import WorkflowStepResult
from design_research_agents._memory._stores._sqlite_store import SQLiteMemoryStore
from design_research_agents.llm import LlamaCppServerLLMClient
from design_research_agents.llm._backends._providers import _llama_cpp_server


def test__llama_cpp_server_command_contains_expected_args() -> None:
    backend = _llama_cpp_server.create_backend(
        model="/tmp/model.gguf",
        hf_model_repo_id="repo/id",
        api_model="local-model",
        host="0.0.0.0",
        port=9000,
        extra_server_args=("--n-gpu-layers", "35"),
    )

    command = backend._build_command()

    assert "llama_cpp.server" in command
    assert "--model" in command
    assert "/tmp/model.gguf" in command
    assert "--model_alias" in command
    assert "local-model" in command
    assert "--hf_model_repo_id" in command
    assert "repo/id" in command
    assert "--n-gpu-layers" in command


def test_llama_cpp_wait_until_ready_retries_after_timeout(
    monkeypatch,
) -> None:
    backend = _llama_cpp_server.LlamaCppServerBackend(
        model="/tmp/model.gguf",
        startup_timeout_seconds=1.0,
        poll_interval_seconds=0.0,
    )

    class _AliveProcess:
        def __init__(self) -> None:
            self._terminated = False

        def poll(self) -> int | None:
            return 0 if self._terminated else None

        def terminate(self) -> None:
            self._terminated = True

        def kill(self) -> None:
            self._terminated = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self._terminated = True
            return 0

    backend._process = _AliveProcess()  # type: ignore[assignment]
    attempts = {"count": 0}

    class _ReadyResponse:
        status = 200

        def __enter__(self) -> _ReadyResponse:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            del exc_type, exc, tb
            return False

    def _flaky_probe(url: str, timeout: float) -> _ReadyResponse:
        del url, timeout
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise TimeoutError("timed out")
        return _ReadyResponse()

    monkeypatch.setattr(_llama_cpp_server, "urlopen", _flaky_probe)
    backend._wait_until_ready()

    assert attempts["count"] == 2


def test_llama_cpp_wait_until_ready_timeout_error_becomes_runtime_error(
    monkeypatch,
) -> None:
    backend = _llama_cpp_server.LlamaCppServerBackend(
        model="/tmp/model.gguf",
        startup_timeout_seconds=0.01,
        poll_interval_seconds=0.0,
    )

    class _AliveProcess:
        def __init__(self) -> None:
            self._terminated = False

        def poll(self) -> int | None:
            return 0 if self._terminated else None

        def terminate(self) -> None:
            self._terminated = True

        def kill(self) -> None:
            self._terminated = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self._terminated = True
            return 0

    backend._process = _AliveProcess()  # type: ignore[assignment]

    def _timeout_probe(url: str, timeout: float) -> object:
        del url, timeout
        raise TimeoutError("timed out")

    monkeypatch.setattr(_llama_cpp_server, "urlopen", _timeout_probe)
    monkeypatch.setattr(_llama_cpp_server.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="Timed out waiting for llama-cpp server readiness"):
        backend._wait_until_ready()


def test__llama_cpp_server_requires_non_empty_model_and_api_model() -> None:
    with pytest.raises(ValueError, match="model must not be empty"):
        _llama_cpp_server.LlamaCppServerBackend(model="   ")

    with pytest.raises(ValueError, match="api_model must not be empty"):
        _llama_cpp_server.LlamaCppServerBackend(model="/tmp/model.gguf", api_model="   ")


def test__llama_cpp_server_dependency_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _llama_cpp_server.LlamaCppServerBackend(model="/tmp/model.gguf")
    monkeypatch.setattr(_llama_cpp_server, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="llama-cpp server dependency is missing"):
        backend._ensure_server_dependency()

    def _find_spec(name: str) -> object | None:
        if name == "llama_cpp.server":
            return object()
        return None

    backend_hf = _llama_cpp_server.LlamaCppServerBackend(model="tiny.Q4_K_M.gguf", hf_model_repo_id="repo/id")
    monkeypatch.setattr(_llama_cpp_server, "find_spec", _find_spec)
    with pytest.raises(RuntimeError, match="huggingface-hub is required"):
        backend_hf._ensure_server_dependency()


def test__llama_cpp_server_resolve_hf_model_name_prefers_unique_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _llama_cpp_server.LlamaCppServerBackend(
        model="Q4_K_M.gguf",
        hf_model_repo_id="repo/id",
    )

    fake_hf = SimpleNamespace(
        list_repo_files=lambda _repo: [
            "model-A.Q4_K_M.gguf",
            "model-A.Q8_0.gguf",
        ]
    )
    monkeypatch.setattr(_llama_cpp_server.importlib, "import_module", lambda _name: fake_hf)

    backend._resolve_hf_model_name()
    assert backend.model == "model-A.Q4_K_M.gguf"


def test_llama_cpp_wait_until_ready_accepts_auth_challenge(monkeypatch: pytest.MonkeyPatch) -> None:
    backend = _llama_cpp_server.LlamaCppServerBackend(
        model="/tmp/model.gguf",
        startup_timeout_seconds=1.0,
        poll_interval_seconds=0.0,
    )

    class _AliveProcess:
        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return 0

    backend._process = _AliveProcess()  # type: ignore[assignment]

    def _auth_probe(_url: str, timeout: float) -> object:
        del timeout
        raise HTTPError(url="http://x", code=401, msg="Unauthorized", hdrs=None, fp=None)

    monkeypatch.setattr(_llama_cpp_server, "urlopen", _auth_probe)
    backend._wait_until_ready()


def test_llama_cpp_close_forces_kill_when_terminate_stalls() -> None:
    backend = _llama_cpp_server.LlamaCppServerBackend(model="/tmp/model.gguf")

    class _StubbornProcess:
        def __init__(self) -> None:
            self.killed = False
            self.wait_calls = 0

        def poll(self) -> int | None:
            return None

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            self.killed = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="llama", timeout=5)
            return 0

    process = _StubbornProcess()
    backend._process = process  # type: ignore[assignment]
    backend.close()
    assert process.killed is True
    assert backend._process is None


def test_llama_cpp_client_constructor_propagates_settings() -> None:
    client = LlamaCppServerLLMClient(
        name="custom-llama",
        model="/tmp/model.gguf",
        hf_model_repo_id="repo/id",
        api_model="custom-model",
        host="0.0.0.0",
        port=9100,
        context_window=2048,
        extra_server_args=("--n-gpu-layers", "35"),
        model_patterns=("custom-model",),
    )
    try:
        assert client.default_model() == "custom-model"
        assert client._backend.name == "custom-llama"
        assert client._backend.model_patterns == ("custom-model",)

        command = client._llama_server._build_command()
        assert "--n_ctx" in command
        assert "2048" in command
        assert "--n-gpu-layers" in command
        assert "35" in command
    finally:
        client.close()


def test_memory_contract_dataclasses_are_serializable() -> None:
    write_record = MemoryWriteRecord(content="draft", metadata={"phase": 1}, item_id="abc")
    search_query = MemorySearchQuery(
        text="draft",
        namespace="design",
        top_k=3,
        min_score=0.25,
        metadata_filters={"phase": 1},
    )
    read_record = MemoryRecord(
        item_id="abc",
        namespace="design",
        content="draft",
        metadata={"phase": 1},
        score=0.9,
        lexical_score=0.8,
        vector_score=0.95,
    )

    assert write_record.to_dict()["item_id"] == "abc"
    assert search_query.to_dict()["top_k"] == 3
    assert read_record.to_dict()["score"] == 0.9


def test_sqlite_memory_store_satisfies_memory_store_protocol(tmp_path) -> None:
    store = SQLiteMemoryStore(db_path=tmp_path / "memory.sqlite3")

    def _accept_memory_store(protocol_value: MemoryStore) -> Sequence[MemoryRecord]:
        protocol_value.write([MemoryWriteRecord(content="hello")], namespace="default")
        return protocol_value.search(MemorySearchQuery(text="hello", namespace="default"))

    matches = _accept_memory_store(store)
    store.close()

    assert len(matches) == 1
    assert matches[0].content == "hello"


def test_execution_result_convenience_accessors_and_to_json() -> None:
    success_result = ExecutionResult(
        success=True,
        output={
            "final_output": {"decision": "approve"},
            "terminated_reason": "completed",
        },
    )
    failure_result = ExecutionResult(
        success=False,
        output={
            "error": {"message": "tool failed"},
            "terminated_reason": "step_failure",
        },
    )

    assert success_result.final_output == {"decision": "approve"}
    assert success_result.terminated_reason == "completed"
    assert success_result.error is None
    assert success_result.output_value("missing", "fallback") == "fallback"
    assert success_result.output_dict("final_output") == {"decision": "approve"}
    assert success_result.output_dict("terminated_reason") == {}
    assert success_result.output_list("events") == []

    assert failure_result.final_output is None
    assert failure_result.terminated_reason == "step_failure"
    assert failure_result.error == {"message": "tool failed"}
    assert failure_result.output_value("error") == {"message": "tool failed"}
    assert failure_result.output_dict("error") == {"message": "tool failed"}
    assert failure_result.output_list("error") == []

    encoded = success_result.to_json()
    assert '"final_output"' in encoded
    assert '"terminated_reason"' in encoded
    assert str(success_result) == encoded


def test_workflow_step_result_convenience_accessors_and_to_dict() -> None:
    result = WorkflowStepResult(
        step_id="finalize",
        status="completed",
        success=True,
        output={
            "final_output": {"decision": "approve"},
            "terminated_reason": "completed",
            "detail": "ok",
        },
    )

    assert result.final_output == {"decision": "approve"}
    assert result.terminated_reason == "completed"
    serialized = result.to_dict()
    assert serialized["step_id"] == "finalize"
    assert serialized["output"]["detail"] == "ok"
