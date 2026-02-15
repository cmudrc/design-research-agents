## Agent Examples

This folder demonstrates the core agent interfaces and execution styles used in
the project.

## Subfolders

- `examples/agents/basic`
  - Non-streaming runs for the major agent types.
  - See `examples/agents/basic/README.md`.
- `examples/agents/streaming`
  - Streaming runs that emit incremental events and completion payloads.
  - See `examples/agents/streaming/README.md`.

## When To Use What

- Use `basic` examples to validate agent behavior and result payload structure.
- Use `streaming` examples to validate event handling (`delta`, `completed`).

## Notes

- Examples are intended to run from repository root with `PYTHONPATH=src`.
- Most agent examples use `LlamaCppServerLLMClient()` by default.
