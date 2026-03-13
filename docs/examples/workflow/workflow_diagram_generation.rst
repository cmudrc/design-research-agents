Workflow Diagram Generation
===========================

Source: ``examples/workflow/workflow_diagram_generation.py``

Introduction
------------

Workflow definitions stay readable while they are small, but nested loops and conditional routing
get harder to inspect once orchestration grows. This example shows how to generate a deterministic
Mermaid diagram directly from a configured ``Workflow`` so the same topology can be reused in local
development, docs, and review discussions.

Technical Implementation
------------------------

1. Build a representative workflow using only the public workflow primitives.
2. Call ``Workflow.to_mermaid(direction="LR")`` to render the declared topology.
3. Persist the Mermaid text under ``artifacts/examples/`` for local inspection or docs reuse.
4. Print a compact JSON payload so example automation can verify the generated diagram shape.

.. mermaid::

   flowchart LR
       A["Workflow definition"] --> B["Workflow.to_mermaid()"]
       B --> C["Mermaid flowchart text"]
       C --> D["artifacts/examples/workflow_diagram.mmd"]
       C --> E["JSON summary for tests/docs"]

.. literalinclude:: ../../../examples/workflow/workflow_diagram_generation.py
   :language: python
   :lines: 45-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python3 examples/workflow/workflow_diagram_generation.py

Example output shape (values vary by run):

.. code-block:: text

   {
     "diagram_path": "artifacts/examples/workflow_diagram.mmd",
     "line_count": 18,
     "starts_with": "flowchart LR",
     "contains_loop": true,
     "contains_route": true
   }

References
----------

- `Mermaid flowcharts <https://mermaid.js.org/syntax/flowchart.html>`_
- `Mermaid subgraphs <https://mermaid.js.org/syntax/flowchart.html#subgraphs>`_
- `Mermaid node shapes and labels <https://mermaid.js.org/syntax/flowchart.html#node-shapes>`_
