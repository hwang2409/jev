import importlib

__all__ = [
    "DENIAL_MARKER",
    "TOOL_RESULT_MAX_BYTES",
    "RuntimeComposition",
    "build_unattended_loop",
    "compose_runtime",
    "drive_turn",
]

_EXPORTS = {
    "DENIAL_MARKER": (".driver", "DENIAL_MARKER"),
    "TOOL_RESULT_MAX_BYTES": (".driver", "TOOL_RESULT_MAX_BYTES"),
    "RuntimeComposition": (".composition", "RuntimeComposition"),
    "build_unattended_loop": (".unattended", "build_unattended_loop"),
    "compose_runtime": (".composition", "compose_runtime"),
    "drive_turn": (".driver", "drive_turn"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attribute_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error
    module = importlib.import_module(module_name, __name__)
    return getattr(module, attribute_name)
