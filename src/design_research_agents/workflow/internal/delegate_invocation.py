"""Re-export canonical delegate invocation helpers for workflow internals."""

from __future__ import annotations

from ...shared.delegate_invocation import (
    DelegateInvocation,
    _invoke_workflow_object_delegate,
    _is_workflow_delegate_runner,
    _is_workflow_object_delegate,
    invoke_delegate,
)

__all__ = [
    "DelegateInvocation",
    "_invoke_workflow_object_delegate",
    "_is_workflow_delegate_runner",
    "_is_workflow_object_delegate",
    "invoke_delegate",
]
