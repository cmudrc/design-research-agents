"""Helpers for optional backend dependency errors."""

from __future__ import annotations

from typing import NoReturn


def raise_missing_optional_dependency(
    *,
    package_name: str,
    extra_name: str,
    feature_name: str,
) -> NoReturn:
    """Raise a standardized optional-dependency installation error.

    Args:
        package_name: Missing importable package name.
        extra_name: Package extra that provides the dependency.
        feature_name: User-facing feature label requiring the dependency.

    Raises:
        RuntimeError: Always raised with an actionable install hint.
    """
    raise RuntimeError(
        f"The '{package_name}' package is required for {feature_name}. "
        f'Install it with: pip install "design-research-agents[{extra_name}]"'
    )
