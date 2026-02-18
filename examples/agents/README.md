## Agent Examples

This folder demonstrates the core agent interfaces and execution styles used in
the project.

## Subfolders

- `examples/agents/basic`
  - Non-streaming runs for the major agent types.
  - See `examples/agents/basic/README.md`.

## When To Use What

- Use `basic` examples to validate agent behavior and result payload structure.

## Notes

- Examples are intended to run from repository root with `PYTHONPATH=src`.
- Most agent examples use `LlamaCppServerLLMClient()` by default.
