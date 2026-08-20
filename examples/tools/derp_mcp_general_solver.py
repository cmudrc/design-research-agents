"""# Tools / DERP MCP General Solver.

## Introduction
This example shows the maintained DRAG + DERP path for selecting a packaged
design-research problem, exposing it as an MCP tool server, and reading solver
hints before making any optimization call. It is intentionally small: use the
DERP catalog API for problem discovery, use the packaged DERP MCP CLI for tool
launch, and keep provider-specific LLM setup outside the problem plumbing.


## Technical Implementation
1. Search DERP with ``search_problem_summaries(...)`` instead of putting full
   problem briefs into the agent prompt.
2. Load the selected optimization problem and read ``solver_hints()`` directly
   so variable domain and constraints do not need to be inferred from prose.
3. Attach the packaged DERP MCP CLI with ``MCPServerConfig.python_module(...)``.
4. Invoke ``solver_hints`` and ``evaluate`` through ``Toolbox`` and print one
   compact JSON payload.

```mermaid
flowchart LR
    A["DERP search_problem_summaries"] --> B["Select problem id"]
    B --> C["DERP solver_hints()"]
    B --> D["python -m design_research_problems.mcp"]
    D --> E["DRAG Toolbox MCP tools"]
    C --> F["Agent-ready routing payload"]
    E --> F
```


## Expected Results
When ``design-research-problems[mcp]`` is available, the example prints a JSON
object containing the selected problem summary, local solver hints, MCP tool
names, MCP solver hints, and one evaluation report. If DERP is not installed,
it exits successfully with an ``available: false`` payload and an install hint.

.. code-block:: text

   {
     "available": true,
     "example": "tools/derp_mcp_general_solver.py",
     "mcp_tools": ["drp_problem::evaluate", "drp_problem::solver_hints", "drp_problem::submit_final"],
     "problem_id": "pill_capsule_min_area"
   }


## References
- `design-research-problems documentation <https://cmudrc.github.io/design-research-problems/>`_
- `Model Context Protocol Specification <https://modelcontextprotocol.io/specification/2025-06-18>`_
- `SciPy optimize documentation <https://docs.scipy.org/doc/scipy/reference/optimize.html>`_
"""

from __future__ import annotations

import json
from typing import Any

import design_research_agents as drag

INSTALL_HINT = 'python -m pip install "design-research-agents[mcp]" "design-research-problems[mcp]"'


def _missing_derp_payload(reason: str) -> dict[str, object]:
    return {
        "available": False,
        "example": "tools/derp_mcp_general_solver.py",
        "reason": reason,
        "install_hint": INSTALL_HINT,
    }


def _load_derp() -> Any | None:
    try:
        import design_research_problems as derp
    except ImportError:
        return None
    required = ("search_problem_summaries", "get_problem", "OptimizationProblem")
    if not all(hasattr(derp, name) for name in required):
        return None
    return derp


def _run_derp_workflow() -> dict[str, object]:
    derp = _load_derp()
    if derp is None:
        return _missing_derp_payload("design-research-problems with catalog summaries is not importable.")

    summaries = derp.search_problem_summaries(text="pill", kind="optimization")
    if not summaries:
        return _missing_derp_payload("No optimization problem summary matched the query.")

    summary = next(
        (candidate for candidate in summaries if candidate.problem_id == "pill_capsule_min_area"),
        summaries[0],
    )
    problem = derp.get_problem(summary.problem_id)
    if not isinstance(problem, derp.OptimizationProblem):
        return _missing_derp_payload(f"Selected problem is not optimization-backed: {summary.problem_id}")

    local_solver_hints = problem.solver_hints()
    initial_candidate = problem.generate_initial_solution(seed=3).tolist()

    try:
        with drag.Toolbox(
            enable_core_tools=False,
            mcp_servers=(
                drag.MCPServerConfig.python_module(
                    id="drp_problem",
                    module="design_research_problems.mcp",
                    args=(summary.problem_id, "--no-citation"),
                    timeout_s=45,
                ),
            ),
        ) as runtime:
            mcp_tools = sorted(spec.name for spec in runtime.list_tools() if spec.name.startswith("drp_problem::"))
            mcp_hints = runtime.invoke(
                "drp_problem::solver_hints",
                {},
                request_id="example-derp-solver-hints",
                dependencies={},
            )
            evaluation = runtime.invoke(
                "drp_problem::evaluate",
                {"x": initial_candidate},
                request_id="example-derp-evaluate",
                dependencies={},
            )
    except Exception as exc:
        return {
            "available": True,
            "example": "tools/derp_mcp_general_solver.py",
            "problem_id": summary.problem_id,
            "problem_summary": summary.to_dict(),
            "local_solver_hints": local_solver_hints,
            "mcp_available": False,
            "mcp_error": str(exc),
            "install_hint": INSTALL_HINT,
        }

    return {
        "available": True,
        "example": "tools/derp_mcp_general_solver.py",
        "problem_id": summary.problem_id,
        "problem_summary": summary.to_dict(),
        "local_solver_hints": local_solver_hints,
        "mcp_tools": mcp_tools,
        "mcp_solver_hints": {
            "ok": mcp_hints.ok,
            "result": mcp_hints.result,
            "error": mcp_hints.error,
        },
        "mcp_evaluation": {
            "ok": evaluation.ok,
            "result": evaluation.result,
            "error": evaluation.error,
        },
    }


def main() -> None:
    """Run the maintained DERP MCP workflow and print JSON."""
    print(json.dumps(_run_derp_workflow(), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
