"""Re-export shared delegate invocation helpers for implementations modules."""

from __future__ import annotations

from ....workflow.internal.delegate_invocation import (
    DelegateInvocation,
    invoke_delegate,
)

__all__ = ["DelegateInvocation", "invoke_delegate"]
