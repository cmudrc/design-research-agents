Prompt Workflow Agent
=====================

Source: ``examples/agents/prompt_workflow_agent.py``

Introduction
------------

This example shows how to package a JSON prompt-mode ``Workflow`` as a reusable study agent for deterministic
design experiments. ``PromptWorkflowAgent`` keeps study prompt construction separate from the workflow that
turns model output into one structured JSON payload.

Technical Implementation
------------------------

1. Define tiny local study packet dataclasses plus a public ``StudyCondition``.
2. Build a prompt-mode ``Workflow`` with ``build_json_prompt_workflow(...)`` and a tiny deterministic LLM
   client.
3. Wrap that workflow in ``PromptWorkflowAgent`` with a prompt builder that converts study metadata into one
   canonical prompt string.
4. Execute an ``AgentRunRequest`` through ``execute_agent_request(...)`` and print a compact JSON payload for
   docs and regression tests.

.. mermaid::

   flowchart LR
       A["Problem packet + run spec + condition"] --> B["PromptWorkflowAgent(prompt_builder)"]
       B --> C["build_json_prompt_workflow"]
       C --> D["json_response"]
       D --> E["JSON payload"]

.. literalinclude:: ../../../examples/agents/prompt_workflow_agent.py
   :language: python
   :lines: 49-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/agents/prompt_workflow_agent.py

Example output shape:

.. code-block:: text

   {
     "workflow_mermaid": "flowchart LR ...",
     "execution": {
       "output": {
         "study_prompt": "Problem: cooling_plate_redesign...",
         "workflow_step": "json_response"
       },
       "metrics": {},
       "event_count": 1
     }
   }

References
----------

- `HELM: Holistic Evaluation of Language Models <https://arxiv.org/abs/2211.09110>`_
- `Python dataclasses <https://docs.python.org/3/library/dataclasses.html>`_
- `Design Research Agents documentation <https://cmudrc.github.io/design-research-agents/>`_
