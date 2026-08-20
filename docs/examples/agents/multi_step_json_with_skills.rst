Multi Step JSON With Skills
===========================

Source: ``examples/agents/multi_step_json_with_skills.py``

Introduction
------------

This example shows a tool-capable multi-step agent with Agent Skills enabled.
Discovered project-local skills can be activated on demand with
``skills.activate`` before the agent selects one of the regular tools.

Technical Implementation
------------------------

1. Build a ``SkillsConfig`` rooted at the current project so the agent can
   discover local ``.agents/skills`` definitions.
2. Construct ``MultiStepAgent`` in JSON mode with the public ``Toolbox`` facade.
3. Allow the model to activate a discovered skill before making a regular tool
   call and explicit final answer.
4. Print the normalized summary payload for inspection.

.. mermaid::

   1. Build a ``SkillsConfig`` rooted at the current project so the agent can
      discover local ``.agents/skills`` definitions.
   2. Construct ``MultiStepAgent`` in JSON mode with the public ``Toolbox`` facade.
   3. Allow the model to activate a discovered skill before making a regular tool
      call and explicit final answer.
   4. Print the normalized summary payload for inspection.

.. literalinclude:: ../../../examples/agents/multi_step_json_with_skills.py
   :language: python
   :lines: 38-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/agents/multi_step_json_with_skills.py

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
- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
- `Toolformer: Language Models Can Teach Themselves to Use Tools <https://arxiv.org/abs/2302.04761>`_
