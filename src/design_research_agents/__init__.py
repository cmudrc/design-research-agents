"""Curated public package interface with lazily resolved top-level exports."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

from design_research_agents._lazy_exports import module_dir, resolve_lazy_export
from design_research_agents._public_exports import (
    TOP_LEVEL_EXPORTS,
    TOP_LEVEL_PUBLIC_API,
    TOP_LEVEL_SUBMODULES,
)

__all__ = list(TOP_LEVEL_PUBLIC_API)

try:
    __version__ = version("design-research-agents")
except PackageNotFoundError:
    __version__ = "0+unknown"


def __getattr__(name: str) -> object:
    """Resolve and cache one deferred public export.

    Args:
        name: Public symbol name requested from the package module.

    Returns:
        Resolved export object.

    Raises:
        AttributeError: If ``name`` is not part of the public export map.
    """
    if name in TOP_LEVEL_SUBMODULES:
        module = import_module(TOP_LEVEL_SUBMODULES[name])
        globals()[name] = module
        return module

    return resolve_lazy_export(
        module_name=__name__,
        exports=TOP_LEVEL_EXPORTS,
        export_name=name,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    """Return package attributes, including deferred exports.

    Returns:
        Sorted attribute list for interactive discovery.
    """
    return module_dir(globals(), __all__)
