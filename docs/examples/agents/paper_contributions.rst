Paper Contributions
===================

Source: ``examples/agents/paper_contributions.py``

Introduction
------------

Agent studies need a compact writing-support boundary that records configuration without
mistaking availability for execution. This offline example describes a tracer before any
run evidence exists, so the resulting packet keeps the missing trace visible as a gap.

Technical Implementation
------------------------

1. Configure a disabled ``Tracer`` without starting an agent run.
2. Call ``collect_agent_paper_contributions(...)`` with a stable component identifier.
3. Verify the public paper-contribution contract version and print the JSON-compatible packet.

.. mermaid::

   flowchart LR
       A["Configured tracer"] --> B["collect_agent_paper_contributions(...)"]
       B --> C["Configured Methods contribution"]
       B --> D["Execution evidence gap"]
       C --> E["Versioned component packet"]
       D --> E

.. literalinclude:: ../../../examples/agents/paper_contributions.py
   :language: python
   :lines: 35-
   :linenos:

Expected Results
----------------

.. rubric:: Run Command

.. code-block:: bash

   PYTHONPATH=src python examples/agents/paper_contributions.py

The printed packet contains a configured Methods contribution, package provenance, and an
``execution-not-provided`` gap. It contains no observed execution claim.

References
----------

- `ReAct: Synergizing Reasoning and Acting in Language Models <https://arxiv.org/abs/2210.03629>`_
- `The ACM Artifact Review and Badging policy <https://www.acm.org/publications/policies/artifact-review-badging>`_
- `FAIR Guiding Principles for scientific data management <https://doi.org/10.1038/sdata.2016.18>`_
