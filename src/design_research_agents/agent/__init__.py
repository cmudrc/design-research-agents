"""Stable public agent facade exports with lazy loading."""

from __future__ import annotations

from typing import Final

from design_research_agents._lazy_exports import module_dir, resolve_lazy_export

_EXPORTS: Final[dict[str, str]] = {
    "DirectLLMCall": "design_research_agents._implementations._agents:DirectLLMCall",
    "MultiStepAgent": "design_research_agents._implementations._agents:MultiStepAgent",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> object:
    """Resolve exported agent symbols on first access.

    Args:
        name: Exported symbol name requested by the caller.

    Returns:
        Resolved exported symbol object.

    Raises:
        AttributeError: Raised when ``name`` is not part of the public exports.
    """
    return resolve_lazy_export(
        module_name=__name__,
        exports=_EXPORTS,
        export_name=name,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    """Return module attributes, including lazy exports.

    Returns:
        Sorted attribute names visible on this module.
    """
    return module_dir(globals(), __all__)
