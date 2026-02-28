"""Helpers for modules that lazily expose curated public symbols."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from importlib import import_module


def module_dir(namespace: Mapping[str, object], exports: Mapping[str, str] | list[str] | tuple[str, ...]) -> list[str]:
    """Return a sorted module attribute listing including declared exports.

    Args:
        namespace: Current module global namespace.
        exports: Declared module exports.

    Returns:
        Sorted attribute names visible on the module.
    """
    return sorted(set(namespace) | set(exports))


def resolve_lazy_export(
    *,
    module_name: str,
    exports: Mapping[str, str],
    export_name: str,
    namespace: MutableMapping[str, object],
) -> object:
    """Resolve and cache one lazily-exported symbol.

    Args:
        module_name: Name of the module exposing lazy exports.
        exports: Mapping of exported names to ``module:attribute`` references.
        export_name: Requested exported symbol.
        namespace: Module globals mapping used for cache writes.

    Returns:
        Resolved export object.

    Raises:
        AttributeError: Raised when ``export_name`` is not part of the export map.
    """
    export_ref = exports.get(export_name)
    if export_ref is None:
        raise AttributeError(f"module '{module_name}' has no attribute '{export_name}'")

    target_module_name, attr_name = export_ref.split(":", maxsplit=1)
    value = getattr(import_module(target_module_name), attr_name)
    namespace[export_name] = value
    return value
