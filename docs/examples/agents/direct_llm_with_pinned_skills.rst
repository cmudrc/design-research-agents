Direct LLM With Pinned Skills
=============================

Source: ``examples/agents/direct_llm_with_pinned_skills.py``

Introduction
------------

This example shows how to preload a trusted project-local skill into a one-shot
direct model call. Pinned skills are useful when you want deterministic,
constructor-scoped behavior without exposing automatic activation.

Technical Implementation
------------------------

1. Build a ``SkillsConfig`` that points at the current project root and pins one
   trusted local skill name.
2. Construct ``DirectLLMCall`` through the public top-level API and pass the
   skills config at construction time.
3. Execute one direct request so the pinned skill is injected as system-context
   before the user prompt.
4. Print the normalized summary payload for inspection.

.. mermaid::

   1. Build a ``SkillsConfig`` that points at the current project root and pins one
      trusted local skill name.
   2. Construct ``DirectLLMCall`` through the public top-level API and pass the
      skills config at construction time.
   3. Execute one direct request so the pinned skill is injected as system-context
      before the user prompt.
   4. Print the normalized summary payload for inspection.

.. literalinclude:: ../../../examples/agents/direct_llm_with_pinned_skills.py
   :language: python
   :lines: 39-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/direct_llm_with_pinned_skills.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "success": true,
     "final_output": "<example-specific payload>",
     "terminated_reason": "<string-or-null>",
     "error": null
   }

References
----------

- `Agent Skills specification <https://agentskills.io/specification>`_
- `Prompting Guide for deterministic system context <https://platform.openai.com/docs/guides/prompt-engineering>`_
- `System prompting patterns for reliable instruction following <https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview>`_
