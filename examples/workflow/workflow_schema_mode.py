"""# Workflow / Workflow Schema Mode.

## Introduction
JSON Schema and function-calling conventions are central for reliable machine-to-machine workflow steps,
while the Responses API anchors current structured request/response patterns. This example illustrates
schema-mode workflow execution where each step contract is explicit and testable.


## Technical Implementation
1. Configure ``Tracer`` with JSONL + console output so each run emits machine-readable traces and lifecycle logs.
2. Build the runtime surface (public APIs only) and execute ``Workflow.run(...)`` with a fixed ``request_id``.
3. Configure and invoke ``Toolbox`` integrations (core/script/MCP/callable) before assembling the final payload.
4. Print a compact JSON payload including ``trace_info`` for deterministic tests and docs examples.

```mermaid
flowchart LR
    A["Input prompt or scenario"] --> B["main(): runtime wiring"]
    B --> C["Workflow.run(...)"]
    C --> D["WorkflowRuntime schedules step graph (LogicStep, ToolStep)"]
    C --> E["Tracer JSONL + console events"]
    D --> F["ExecutionResult/payload"]
    E --> F
    F --> G["Printed JSON output"]
```


## Expected Results

Example output shape (values vary by run):

.. code-block:: text

   {
     "strict_run": {
       "success": true,
       "final_output": "<example-specific payload>",
       "terminated_reason": "<string-or-null>",
       "error": null,
       "trace": {
         "request_id": "<request-id>",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
       }
     },
     "relaxed_run": {
       "success": true,
       "final_output": "<example-specific payload>",
       "terminated_reason": "<string-or-null>",
       "error": null,
       "trace": {
         "request_id": "<request-id>",
         "trace_dir": "artifacts/examples/traces",
         "trace_path": "artifacts/examples/traces/run_<timestamp>_<request_id>.jsonl"
       }
     }
   }

## References
- `JSON Schema Draft 2020-12 <https://json-schema.org/draft/2020-12>`_
- `OpenAI Function Calling Guide <https://platform.openai.com/docs/guides/function-calling>`_
- `OpenAI Responses API <https://platform.openai.com/docs/api-reference/responses>`_
"""

from __future__ import annotations

import json
from pathlib import Path

from design_research_agents import ExecutionResult, LogicStep, Toolbox, ToolStep, Tracer, Workflow

INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "required": [
        "dataset_csv_path",
        "quality_report_path",
        "required_columns",
        "sample_nrows",
        "max_missing_ratio_per_column",
    ],
    "properties": {
        "dataset_csv_path": {"type": "string"},
        "quality_report_path": {"type": "string"},
        "required_columns": {"type": "array", "items": {"type": "string"}},
        "sample_nrows": {"type": "integer"},
        "max_missing_ratio_per_column": {"type": "number"},
    },
    "additionalProperties": False,
}


def _summarize(result: ExecutionResult) -> dict[str, object]:
    return result.summary()


def main() -> None:
    """Run schema-mode workflow with strict and relaxed quality thresholds."""
    tracer = Tracer(
        enabled=True,
        trace_dir=Path("artifacts/examples/traces"),
        enable_jsonl=True,
        enable_console=True,
    )
    dataset_path = Path("artifacts/examples/design_schema_dataset.csv")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "\n".join(
            [
                "component_id,variant,serviceability_score,notes",
                "C001,A,4.2,Quick access screws",
                "C002,A,3.8,",
                "C003,B,,Needs gasket redesign",
                "C004,B,4.9,Tool-less latch",
                "C005,A,2.7,Cable route is cramped",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    workflow = Workflow(
        tool_runtime=Toolbox(),
        tracer=tracer,
        steps=[
            ToolStep(
                step_id="describe_dataset",
                tool_name="data.describe",
                input_builder=lambda context: {
                    "path": context["inputs"]["dataset_csv_path"],
                    "kind": "csv",
                },
            ),
            ToolStep(
                step_id="load_sample",
                tool_name="data.load_csv",
                dependencies=("describe_dataset",),
                input_builder=lambda context: {
                    "path": context["inputs"]["dataset_csv_path"],
                    "nrows": context["inputs"]["sample_nrows"],
                },
            ),
            LogicStep(
                step_id="quality_gate",
                dependencies=("describe_dataset", "load_sample"),
                handler=lambda context: {
                    "row_count": (context["dependency_results"]["describe_dataset"]["output"]["result"]["rows"]),
                    "sample_count": (context["dependency_results"]["load_sample"]["output"]["result"]["count"]),
                    "required_columns": context["inputs"]["required_columns"],
                    "threshold": context["inputs"]["max_missing_ratio_per_column"],
                },
            ),
            ToolStep(
                step_id="persist_report",
                tool_name="fs.write_text",
                dependencies=("quality_gate",),
                input_builder=lambda context: {
                    "path": context["inputs"]["quality_report_path"],
                    "content": str(context["dependency_results"]["quality_gate"]["output"]) + "\n",
                    "overwrite": True,
                },
            ),
            LogicStep(
                step_id="finalize",
                dependencies=("persist_report",),
                handler=lambda context: {
                    "report_path": (context["dependency_results"]["persist_report"]["output"]["result"]["path"])
                },
            ),
        ],
        input_schema=INPUT_SCHEMA,
    )

    # Use explicit strict and relaxed ids so each policy run is traceable independently.
    strict_request_id = "example-workflow-schema-design-strict-001"
    strict_result = workflow.run(
        {
            "dataset_csv_path": str(dataset_path),
            "required_columns": ["component_id", "variant", "serviceability_score", "notes"],
            "sample_nrows": 3,
            "quality_report_path": "artifacts/examples/design_schema_quality_strict.txt",
            "max_missing_ratio_per_column": 0.2,
        },
        execution_mode="sequential",
        request_id=strict_request_id,
    )

    # Use explicit strict and relaxed ids so each policy run is traceable independently.
    relaxed_request_id = "example-workflow-schema-design-relaxed-001"
    relaxed_result = workflow.run(
        {
            "dataset_csv_path": str(dataset_path),
            "required_columns": ["component_id", "variant", "serviceability_score", "notes"],
            "sample_nrows": 5,
            "quality_report_path": "artifacts/examples/design_schema_quality_relaxed.txt",
            "max_missing_ratio_per_column": 0.45,
        },
        execution_mode="dag",
        request_id=relaxed_request_id,
    )

    print(
        json.dumps(
            {
                "strict_run": _summarize(strict_result),
                "relaxed_run": _summarize(relaxed_result),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
