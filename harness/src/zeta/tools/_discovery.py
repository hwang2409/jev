"""Discover and register built-in tool modules."""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .registry import ToolRegistry


def discover_tool_modules() -> list[str]:
    package = importlib.import_module(__package__)
    modules = (
        f"{package.__name__}.{module_info.name}"
        for module_info in pkgutil.iter_modules(package.__path__)
        if not module_info.name.startswith("_")
    )
    return sorted(modules, key=lambda name: (name.endswith(".agent"), name))


def register_discovered_tools(registry: ToolRegistry) -> None:
    """Load modules with ``register(registry)``; underscore modules are helpers."""

    for module_name in discover_tool_modules():
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load tool module {module_name}: {exc}"
            ) from exc
        if not hasattr(module, "register"):
            continue
        register = module.register
        if not callable(register):
            raise TypeError(
                f"tool module {module_name} has a non-callable register contract"
            )
        try:
            result = register(registry)
            if inspect.isawaitable(result):
                if inspect.iscoroutine(result):
                    result.close()
                raise TypeError("register must be synchronous")
        except Exception as exc:
            raise RuntimeError(
                f"failed to register tool module {module_name}: {exc}"
            ) from exc


_discover_tool_modules = discover_tool_modules
_register_discovered_tools = register_discovered_tools
