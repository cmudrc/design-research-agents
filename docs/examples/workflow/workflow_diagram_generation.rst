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
3. Call ``Workflow.to_svg(direction="LR")`` to emit a static SVG for notebooks, docs assets, or reviews.
4. Persist both diagram formats under ``artifacts/examples/`` for local inspection or docs reuse.
5. Print a compact JSON payload so example automation can verify the generated diagram shape.

The diagram below is generated from the example's configured ``Workflow``.

.. mermaid::

   flowchart LR
       workflow_entry["Workflow Entrypoint"]
       step_1["prepare<br/>LogicStep"]
       step_2["review_loop<br/>LoopStep<br/>max_iterations=2"]
       step_3["publish<br/>LogicStep"]
       subgraph loop_body_1["Loop Body: review_loop"]
           direction TD
           loop_entry_1["review_loop iteration entry"]
           step_4["review_loop::router<br/>LogicStep"]
           step_5["review_loop::draft<br/>LogicStep"]
           step_6["review_loop::score<br/>LogicStep"]
           loop_entry_1 --> step_4
           step_4 -. "route=draft_path" .-> step_5
           step_4 -. "route=score_path" .-> step_6
           step_4 --> step_5
           step_5 --> step_6
           step_6 -. "next iteration" .-> loop_entry_1
       end
       workflow_entry --> step_1
       step_1 --> step_2
       step_2 -. "iterate" .-> loop_entry_1
       step_2 --> step_3

.. literalinclude:: ../../../examples/workflow/workflow_diagram_generation.py
   :language: python
   :lines: 40-
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
     "svg_path": "artifacts/examples/workflow_diagram.svg",
     "line_count": 18,
     "starts_with": "flowchart LR",
     "svg_starts_with": "<svg",
     "contains_loop": true,
     "contains_route": true
   }

References
----------

- `Mermaid flowcharts <https://mermaid.js.org/syntax/flowchart.html>`_
- `Mermaid subgraphs <https://mermaid.js.org/syntax/flowchart.html#subgraphs>`_
- `Mermaid node shapes and labels <https://mermaid.js.org/syntax/flowchart.html#node-shapes>`_
