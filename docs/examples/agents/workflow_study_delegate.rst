Workflow Study Delegate
=======================

Source: ``examples/agents/workflow_study_delegate.py``

Introduction
------------

This example shows how ``WorkflowStudyDelegate`` can wrap a prompt-mode workflow for
packaged-problem-style study execution without introducing external dependencies. The
delegate uses structured study metadata to build the workflow prompt while keeping the
runtime surface the same ``run(prompt, dependencies=...)`` contract used elsewhere in
the public API.

Technical Implementation
------------------------

1. Define tiny local study packet stubs for the problem, run, and condition inputs.
2. Build a prompt-mode ``Workflow`` that echoes the resolved study prompt and metadata.
3. Wrap that workflow with ``WorkflowStudyDelegate`` and a deterministic prompt builder.
4. Print a small JSON payload showing the compiled prompt and workflow output.

.. mermaid::

   flowchart LR
       A["Problem packet"] --> D["WorkflowStudyDelegate"]
       B["Run spec"] --> D
       C["Condition"] --> D
       D --> E["Prompt-mode Workflow"]
       E --> F["JSON study output"]

.. literalinclude:: ../../../examples/agents/workflow_study_delegate.py
   :language: python
   :lines: 47-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/agents/workflow_study_delegate.py

Example output shape:

.. code-block:: text

   {
     "compiled_input": "Study local heat sink concept alternatives under cond-baseline for run-3.",
     "final_output": {
       "condition_id": "cond-baseline",
       "problem_brief": "Compare local heat sink concept alternatives.",
       "prompt": "Study local heat sink concept alternatives under cond-baseline for run-3.",
       "run_id": "run-3"
     }
   }

References
----------

- `Workflow examples <https://cmudrc.github.io/design-research-agents/examples/workflow/index.html>`_
- `Pattern overview <https://cmudrc.github.io/design-research-agents/patterns/overview.html>`_
- `Design Research Agents documentation <https://cmudrc.github.io/design-research-agents/>`_
