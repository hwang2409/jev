import importlib
import sys
import types

__all__ = ["build_parser", "main"]

_EXPORTS = {
    "build_parser": (".main", "build_parser"),
    "main": (".main", "main"),
    "create_app": (".main", "create_app"),
    "_cleanup_ephemeral": (".main", "_cleanup_ephemeral"),
    "_print_exit_hint": (".main", "_print_exit_hint"),
    "_run_login": (".main", "_run_login"),
}


class _CliModule(types.ModuleType):
    def __setattr__(self, name: str, value: object) -> None:
        if (
            name == "main"
            and isinstance(value, types.ModuleType)
            and value.__name__ == f"{__name__}.main"
        ):
            value = value.main
        super().__setattr__(name, value)


sys.modules[__name__].__class__ = _CliModule


def __getattr__(name: str) -> object:
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
