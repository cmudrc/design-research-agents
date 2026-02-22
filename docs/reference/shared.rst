Shared Modules
==============

The legacy ``design_research_agents._shared`` example helper modules were
removed.

Runnable examples are now capability-first and helper-free. Deterministic
fixtures for tests live only in
``tests/example_monkeypatch/sitecustomize.py`` and are enabled by setting
``DRA_EXAMPLE_LLM_MODE=deterministic`` in test runs.
