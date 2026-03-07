"""Internal shared helpers for workflow-native pattern implementations."""

from design_research_agents._runtime._common._run_defaults import (
    normalize_request_id_prefix,
)

from ._batch_results import (
    extract_call_error,
    extract_call_model_response,
    extract_call_output,
    extract_delegate_batch_call_result,
    extract_delegate_batch_call_result_from_context,
    is_call_success,
)
from ._loop_callbacks import (
    LoopCallbacks,
    LoopIterationHandler,
    build_loop_callbacks,
    wrap_iteration_handler,
)
from ._loop_state import normalize_mapping, normalize_mapping_records, parse_loop_iteration
from ._pattern_contract import (
    MODE_BLACKBOARD,
    MODE_DEBATE,
    MODE_PLAN_EXECUTE,
    MODE_PROPOSE_CRITIC,
    MODE_RAG,
    MODE_RALPH_LOOP,
    MODE_ROUND_BASED_COORDINATION,
    MODE_ROUTER_DELEGATE,
    MODE_TREE_SEARCH,
    MODE_TWO_SPEAKER_CONVERSATION,
    OUTPUT_CONTRACT_VERSION,
    build_pattern_execution_result,
    build_pattern_output,
)
from ._prompt_rendering import render_prompt_template, resolve_prompt_override
from ._run_context import (
    PatternRunContext,
    WorkflowBudgetTracker,
    attach_pattern_workflow,
    attach_runtime_metadata,
    build_compiled_pattern_execution,
    build_pattern_failure_result,
    build_workflow_output_payload,
    execute_pattern_with_trace,
    resolve_pattern_run_context,
)

__all__ = [
    "MODE_BLACKBOARD",
    "MODE_DEBATE",
    "MODE_PLAN_EXECUTE",
    "MODE_PROPOSE_CRITIC",
    "MODE_RAG",
    "MODE_RALPH_LOOP",
    "MODE_ROUND_BASED_COORDINATION",
    "MODE_ROUTER_DELEGATE",
    "MODE_TREE_SEARCH",
    "MODE_TWO_SPEAKER_CONVERSATION",
    "OUTPUT_CONTRACT_VERSION",
    "LoopCallbacks",
    "LoopIterationHandler",
    "PatternRunContext",
    "WorkflowBudgetTracker",
    "attach_pattern_workflow",
    "attach_runtime_metadata",
    "build_compiled_pattern_execution",
    "build_loop_callbacks",
    "build_pattern_execution_result",
    "build_pattern_failure_result",
    "build_pattern_output",
    "build_workflow_output_payload",
    "execute_pattern_with_trace",
    "extract_call_error",
    "extract_call_model_response",
    "extract_call_output",
    "extract_delegate_batch_call_result",
    "extract_delegate_batch_call_result_from_context",
    "is_call_success",
    "normalize_mapping",
    "normalize_mapping_records",
    "normalize_request_id_prefix",
    "parse_loop_iteration",
    "render_prompt_template",
    "resolve_pattern_run_context",
    "resolve_prompt_override",
    "wrap_iteration_handler",
]
