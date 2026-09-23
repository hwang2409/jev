"""Interactive terminal interface for zeta."""

import importlib

__all__ = ["TUIApp", "main"]

_EXPORTS = {
    "TUIApp": (".app", "TUIApp"),
}


def __getattr__(name: str) -> object:
    if name == "main":
        from ..cli import main

        return main
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value
