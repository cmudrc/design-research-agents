Agent Examples
==============

These scripts demonstrate traced agent entrypoints for engineering-design
prompting and tool orchestration.

Scripts and observations
------------------------

- ``examples/agents/basic/direct_llm_call.py``
  Observe: one-shot output in ``final_output`` and a non-empty ``trace.trace_path``.
  Public API: ``DirectLLMCall``, ``LlamaCppServerLLMClient``, ``__version__``.
- ``examples/agents/basic/multi_step_direct_llm_agent.py``
  Observe: controller progression via ``steps_executed`` and ``terminated_reason``.
  Public API: ``MultiStepAgent``.
- ``examples/agents/basic/multi_step_json_tool_calling_agent.py``
  Observe: tool loop summary and ``tool_results_count``.
  Public API: ``CallableTool``, ``Toolbox``.
- ``examples/agents/basic/multi_step_code_tool_calling_agent.py``
  Observe: non-zero ``step_outputs_count`` and tool usage traces.
  Public API: ``MultiStepAgent``, ``Toolbox``.
- ``examples/agents/basic/multi_step_json_with_memory.py``
  Observe: memory retrieval/write participation in ``memory_items``.
  Public API: ``MultiStepAgent``, ``Toolbox``.

Observed local run snippets (2026-02-21)
----------------------------------------

``examples/agents/basic/direct_llm_call.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/agents/basic/direct_llm_call.py

Observed stdout:

.. code-block:: json

   {
     "example": "agents/basic/direct_llm_call.py",
     "final_output": "4",
     "success": true,
     "trace": {
       "request_id": "example-direct-llm-design-001",
       "trace_path": "artifacts/examples/traces/run_<timestamp>_example-direct-llm-design-001.jsonl"
     }
   }

Observed trace log excerpt (from the emitted JSONL):

.. code-block:: json

   {"event_type":"RunStarted","run_id":"example-direct-llm-design-001","attributes":{"agent":"DirectLLMCall"}}
   {"event_type":"ModelCallFinished","run_id":"example-direct-llm-design-001","attributes":{"model":"example-model"}}

``examples/agents/basic/multi_step_direct_llm_agent.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/agents/basic/multi_step_direct_llm_agent.py

Observed stdout:

.. code-block:: json

   {
     "example": "agents/basic/multi_step_direct_llm_agent.py",
     "final_output": "42",
     "steps_executed": 2,
     "success": true,
     "terminated_reason": "stop:model"
   }

``examples/agents/basic/multi_step_json_with_memory.py``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run command:

.. code-block:: bash

   PYTHONPATH=tests/example_monkeypatch:src \
   DRA_EXAMPLE_LLM_MODE=deterministic \
   python3 examples/agents/basic/multi_step_json_with_memory.py

Observed stdout:

.. code-block:: json

   {
     "example": "agents/basic/multi_step_json_with_memory.py",
     "final_output": {"expression": "12 * (4 + 1)", "result": 60.0},
     "memory_items": 5,
     "tool_results_count": 1,
     "success": true,
     "terminated_reason": "continuation_stopped:model"
   }
