Prompt Workflow Agent
=====================

Source: ``examples/agents/prompt_workflow_agent.py``

Introduction
------------

This example shows how to package a prompt-mode ``Workflow`` as a reusable study agent for deterministic
design experiments. ``PromptWorkflowAgent`` keeps the workflow itself simple while moving packaged-problem,
run-spec, and condition formatting into one explicit prompt builder.

Technical Implementation
------------------------

1. Define tiny local study packet dataclasses so the example stays dependency-light and deterministic.
2. Build a prompt-mode ``Workflow`` with logic steps that capture the resolved study prompt and emit one final
   summary payload.
3. Wrap that workflow in ``PromptWorkflowAgent`` with a prompt builder that converts study metadata into one
   canonical prompt string.
4. Run the delegate with a fixed ``request_id`` and print a compact JSON payload for docs and regression tests.

.. mermaid::

   flowchart LR
       A["Problem packet + run spec + condition"] --> B["PromptWorkflowAgent(prompt_builder)"]
       B --> C["Prompt-mode Workflow"]
       C --> D["capture_study_prompt"]
       D --> E["emit_summary"]
       E --> F["JSON payload"]

.. literalinclude:: ../../../examples/agents/prompt_workflow_agent.py
   :language: python
   :lines: 54-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/prompt_workflow_agent.py

Example output shape:

.. code-block:: text

   {
     "workflow_mermaid": "flowchart LR ...",
     "summary": {
       "success": true,
       "final_output": {
         "request_id": "example-prompt-workflow-agent-001",
         "study_prompt": "Problem: cooling_plate_redesign...",
         "workflow_step": "emit_summary"
       },
       "terminated_reason": null,
       "error": null,
       "trace": {
         "request_id": "example-prompt-workflow-agent-001"
       }
     }
   }

References
----------

- `HELM: Holistic Evaluation of Language Models <https://arxiv.org/abs/2211.09110>`_
- `Python dataclasses <https://docs.python.org/3/library/dataclasses.html>`_
- `Design Research Agents documentation <https://cmudrc.github.io/design-research-agents/>`_
