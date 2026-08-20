# Examples

Runnable examples for `design-research-agents`, grouped by public subsystem.
All checked-in Python and shell examples are inventoried below.

Examples use public APIs and emit inspectable output. Runtime-oriented examples
usually write trace metadata under `artifacts/examples/traces`; lightweight
onboarding, inspection, helper, and diagram-generation examples may not create a
trace artifact.

## Offline Quickstart

Run the base-install example from repository root. It uses a deterministic local
stub and needs no API key, model download, or model server.

```bash
PYTHONPATH=src python examples/agents/vscode_hello_world.py
```

## Complete Inventory

### Agents

- `examples/agents/vscode_hello_world.py`: offline deterministic first run with `DirectLLMCall`.
- `examples/agents/direct_llm_call.py`: one call to a reachable OpenAI-compatible HTTP endpoint.
- `examples/agents/direct_llm_compiled_execution.py`: compile and inspect a direct-call execution.
- `examples/agents/direct_llm_with_pinned_skills.py`: direct execution with pinned skills context.
- `examples/agents/multi_step_direct_llm_agent.py`: iterative direct-mode execution.
- `examples/agents/multi_step_json_tool_calling_agent.py`: JSON action/observation tool loop.
- `examples/agents/multi_step_json_with_memory.py`: multi-step JSON execution with memory.
- `examples/agents/multi_step_json_with_skills.py`: multi-step JSON execution with skills.
- `examples/agents/multi_step_code_tool_calling_agent.py`: code-action tool loop.
- `examples/agents/seeded_random_baseline_agent.py`: deterministic study control participant.
- `examples/agents/prompt_workflow_agent.py`: packaged-problem prompt workflow participant.

### Workflows

- `examples/workflow/workflow_runtime.py`: core workflow runtime.
- `examples/workflow/workflow_prompt_mode.py`: prompt-mode workflow.
- `examples/workflow/workflow_schema_mode.py`: schema-constrained workflow.
- `examples/workflow/workflow_runtime_loop_step.py`: loop-step execution.
- `examples/workflow/workflow_delegate_and_memory_steps.py`: delegate and memory steps.
- `examples/workflow/workflow_model_step_design_tradeoff.py`: model step for a design tradeoff.
- `examples/workflow/workflow_diagram_generation.py`: workflow diagram generation.

### Patterns

- `examples/patterns/plan_execute.py`: plan/execute orchestration.
- `examples/patterns/propose_critic.py`: proposer/critic iteration.
- `examples/patterns/debate_pattern.py`: multi-role debate and adjudication.
- `examples/patterns/coordination_patterns.py`: round-based and blackboard coordination.
- `examples/patterns/nominal_team.py`: nominal-team workflow.
- `examples/patterns/ralph_loop.py`: Ralph-loop refinement.
- `examples/patterns/router_delegate.py`: routing among delegates.
- `examples/patterns/tree_search.py`: tree-search orchestration.
- `examples/patterns/simulated_annealing.py`: simulated-annealing pattern and schedules.
- `examples/patterns/rag.py`: retrieval-augmented generation.
- `examples/patterns/two_speaker_conversation.py`: two-participant conversation.

### Clients

- `examples/clients/openai_compatible_http_client.py`: built-in HTTP client for a reachable compatible endpoint.
- `examples/clients/openai_service_client.py`: OpenAI hosted-service client.
- `examples/clients/anthropic_service_client.py`: Anthropic hosted-service client.
- `examples/clients/gemini_service_client.py`: Gemini hosted-service client.
- `examples/clients/groq_service_client.py`: Groq hosted-service client.
- `examples/clients/llama_cpp_server_client.py`: managed llama.cpp server client.
- `examples/clients/ollama_local_client.py`: local Ollama client.
- `examples/clients/transformers_local_client.py`: in-process Transformers client.
- `examples/clients/mlx_local_client.py`: Apple MLX client.
- `examples/clients/vllm_server_client.py`: vLLM server client.
- `examples/clients/sglang_server_client.py`: SGLang server client.

### Model Selection

- `examples/model_selection/local.py`: local-backend selection policy.
- `examples/model_selection/remote.py`: hosted-backend selection policy.

### Tools

- `examples/tools/multi_source_tool_usage.py`: callable, script, and MCP tool sources.
- `examples/tools/mcp_minimal.py`: packaged local MCP stdio server.
- `examples/tools/derp_mcp_general_solver.py`: `design-research-problems` MCP solver path.
- `examples/tools/script_tools/repo_quickscan.sh`: shell script-tool repository scan.
- `examples/tools/script_tools/rubric_score.py`: script-tool rubric scorer.

### Optimization

- `examples/optimization/multi_step_json_tool_calling_1d_optimization.py`: one-dimensional optimization through a JSON tool loop.

## Backend Prerequisites

Provider examples require their named extra and credentials. Server-backed
examples require their named extra plus a reachable or managed service. Several
`LlamaCppServerLLMClient` examples intentionally use `Qwen3-4B` GGUF configs;
on lower-memory machines, use a smaller model or start with
`examples/clients/ollama_local_client.py`, which documents the lighter
`qwen2.5:1.5b-instruct` default.

The DERP MCP solver example needs only the MCP extras:

```bash
python -m pip install "design-research-agents[mcp]" "design-research-problems[mcp]"
```

## Deterministic Testing

Run the smoke set with `make examples-smoke`. Full example tests inject
deterministic behavior through `tests/example_monkeypatch/sitecustomize.py` when
`DRA_EXAMPLE_LLM_MODE=deterministic` is set; production examples themselves do
not branch on that test mode.
