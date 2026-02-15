"""Runnable example for the reusable pure-tool workflow orchestration chunk."""

import json
from pathlib import Path

from design_research_agents import PureToolWorkflow, UnifiedToolRuntime
from design_research_agents.contracts.workflow import LogicStep, ToolStep

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


def main() -> None:
    """Run the configured pure-tool workflow twice with different input overrides."""
    dataset_path = Path("artifacts/examples/pure_tool_workflow_dataset.csv")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path.write_text(
        "\n".join(
            [
                "participant_id,study_arm,satisfaction_score,notes",
                "P001,A,4.2,Onboarding clear",
                "P002,A,3.8,",
                "P003,B,,Needed more examples",
                "P004,B,4.9,Very helpful",
                "P005,A,2.7,Confusing navigation",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    workflow_steps = [
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
                "row_count": context["dependency_results"]["describe_dataset"]["output"]["result"][
                    "rows"
                ],
                "sample_count": context["dependency_results"]["load_sample"]["output"]["result"][
                    "count"
                ],
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
                "content": json.dumps(
                    context["dependency_results"]["quality_gate"]["output"],
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                "overwrite": True,
            },
        ),
        LogicStep(
            step_id="finalize",
            dependencies=("persist_report",),
            handler=lambda context: {
                "report_path": context["dependency_results"]["persist_report"]["output"]["result"][
                    "path"
                ]
            },
        ),
    ]

    workflow = PureToolWorkflow(
        tool_runtime=UnifiedToolRuntime(),
        steps=workflow_steps,
        input_schema=INPUT_SCHEMA,
    )

    strict_result = workflow.run(
        inputs={
            "dataset_csv_path": str(dataset_path),
            "required_columns": [
                "participant_id",
                "study_arm",
                "satisfaction_score",
                "notes",
            ],
            "sample_nrows": 3,
            "quality_report_path": "artifacts/examples/pure_tool_workflow_quality_strict.json",
            "max_missing_ratio_per_column": 0.2,
        },
        execution_mode="sequential",
        request_id="example-pure-tool-workflow-strict",
    )
    relaxed_result = workflow.run(
        inputs={
            "dataset_csv_path": str(dataset_path),
            "required_columns": [
                "participant_id",
                "study_arm",
                "satisfaction_score",
                "notes",
            ],
            "sample_nrows": 5,
            "quality_report_path": "artifacts/examples/pure_tool_workflow_quality_relaxed.json",
            "max_missing_ratio_per_column": 0.45,
        },
        execution_mode="dag",
        request_id="example-pure-tool-workflow-relaxed",
    )

    print(
        json.dumps(
            {
                "strict_run": strict_result.asdict(),
                "relaxed_run": relaxed_result.asdict(),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
