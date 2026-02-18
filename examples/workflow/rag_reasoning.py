"""Runnable example for ``RagReasoningPattern`` workflow orchestration."""

from __future__ import annotations

from collections.abc import Mapping

from design_research_agents.contracts.agent import Agent, ExecutionResult
from design_research_agents.contracts.memory import MemoryWriteRecord
from design_research_agents.memory.stores.sqlite_store import SQLiteMemoryStore
from design_research_agents.workflow import RagReasoningPattern


class EchoReasoningAgent(Agent):
    """Deterministic reasoning delegate used for local RAG example."""

    def run(
        self,
        prompt: str,
        *,
        request_id: str | None = None,
        dependencies: Mapping[str, object] | None = None,
    ) -> ExecutionResult:
        """Return a deterministic reasoning payload for the given prompt.

        Args:
            prompt: Reasoning prompt built from task and retrieved context.
            request_id: Optional request identifier.
            dependencies: Optional dependency mapping.

        Returns:
            Deterministic reasoning result.
        """
        del request_id, dependencies
        return ExecutionResult(
            output={
                "summary": "Reasoned with retrieved context.",
                "prompt_seen": prompt,
            },
            success=True,
            tool_results=[],
            model_response=None,
            metadata={"delegate": "echo"},
        )


def main() -> None:
    """Run one local RAG workflow and print result."""
    store = SQLiteMemoryStore()
    store.write(
        [
            MemoryWriteRecord(
                content="Design requirement: include graceful shutdown and monitoring.",
                metadata={"kind": "requirement"},
            )
        ],
        namespace="examples",
    )

    pattern = RagReasoningPattern(
        reasoning_delegate=EchoReasoningAgent(),
        memory_store=store,
        memory_namespace="examples",
        memory_top_k=3,
        write_back=True,
    )
    result = pattern.run("Draft a concise architecture recommendation.")
    print(result.asdict())
    store.close()


if __name__ == "__main__":
    main()
