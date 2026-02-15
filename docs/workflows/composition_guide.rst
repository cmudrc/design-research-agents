Composition Guide
=================

Use workflows as composable building blocks, not fixed pipelines.

Structure and extension points
------------------------------

- Configure agents, tools, and step topology at initialization.
- Reuse the same workflow instance across multiple runs.
- Isolate scenario-specific behavior in step logic.
- Combine deterministic logic with model-driven decisions where needed.

Observability and bells and whistles
------------------------------------

- Emit traces for debugging and evaluation.
- Capture intermediate step outputs for auditability.
- Keep failure policy explicit per workflow.
- Use model selection constraints when workflows span local/remote backends.

Background references
---------------------

- `Plan-and-Solve Prompting <https://arxiv.org/abs/2305.04091>`_
- `Reflexion <https://arxiv.org/abs/2303.11366>`_
- `Tree of Thoughts <https://arxiv.org/abs/2305.10601>`_
