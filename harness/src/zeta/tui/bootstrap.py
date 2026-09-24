"""TUI-specific session startup glue around the shared runtime bootstrap."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING

from ..core.session import (
    SessionError,
    SessionManager,
    SessionPreview,
    env_home,
)
from ..runtime.bootstrap import (
    RECENT_SESSION_LIMIT,
    create_runtime_bootstrap,
    format_picker_row,
    pick_resume_session,
)
from . import theme as _theme
from .key_bindings import KeybindingError, resolve_keybindings
from .layout import content_width, resume_picker_line

if TYPE_CHECKING:
    from .app import TUIApp

def create_app(args: argparse.Namespace) -> TUIApp:
    home = env_home()
    ephemeral_root = None
    if bool(getattr(args, "no_session", False)):
        import tempfile

        ephemeral_root = Path(tempfile.mkdtemp(prefix="zeta-ephemeral-"))
    try:
        with ExitStack() as cleanup:
            app = _create_app_with_root(args, home, ephemeral_root, cleanup)
            cleanup.pop_all()
            return app
    except BaseException:
        if ephemeral_root is not None:
            import shutil

            shutil.rmtree(ephemeral_root, ignore_errors=True)
        raise


def _create_app_with_root(
    args: argparse.Namespace,
    home: Path,
    ephemeral_root: Path | None,
    cleanup: ExitStack,
) -> TUIApp:
    from . import app as _app

    runtime = create_runtime_bootstrap(
        args,
        home=home,
        ephemeral_root=ephemeral_root,
        cleanup=cleanup,
        resume_picker=lambda manager: _pick_resume_session(manager, _app),
        load_settings_fn=_app.load_settings,
        load_project_context_fn=_app.load_project_context,
        backend_builder=_app.build_backend,
        persist_plan_mode=True,
    )
    composition = runtime.composition
    theme_notices = _apply_startup_theme(runtime.config.theme, home)
    _validate_keybindings(runtime.config.keybindings)
    startup_notices = (
        tuple(runtime.loaded_settings.notices)
        + runtime.project_context.notices
        + composition.external_tools.notices
        + theme_notices
    )
    startup_warnings = (
        tuple(runtime.loaded_settings.warnings) + composition.external_tools.warnings
    )
    if ephemeral_root is not None:
        startup_warnings = (
            "ephemeral session: nothing will be persisted",
            *startup_warnings,
        )
    startup_alerts = (
        ("system prompt overridden for this session; prompt cache will rebuild",)
        if runtime.resuming and runtime.override_on_resume
        else ()
    )
    return _app.TUIApp(
        composition.loop,
        provider=runtime.provider,
        model=composition.model,
        zeta_home=home,
        verbose=args.verbose,
        history_path=(ephemeral_root if ephemeral_root is not None else home)
        / "history",
        approval_policy=composition.policy,
        context_files=[str(path) for path in runtime.project_context.files],
        on_model_change=runtime.on_model_change,
        vim_mode=runtime.metadata.vim_mode,
        on_budget_change=(
            None
            if composition.budget_pinned
            else lambda budget: runtime.manager.record_budget(
                runtime.metadata, budget=budget, pinned=False
            )
        ),
        on_vim_mode_change=lambda enabled: runtime.manager.record_vim_mode(
            runtime.metadata, enabled=enabled
        ),
        startup_notices=startup_notices,
        startup_warnings=startup_warnings,
        startup_alerts=startup_alerts,
        external_tools=composition.external_tools,
        workspace_snapshot_cap=runtime.config.workspace_snapshot_cap,
        ephemeral_root=ephemeral_root,
        session_name=runtime.metadata.name,
        on_name_change=lambda label: runtime.manager.record_name(
            runtime.metadata, name=label
        ),
        key_remap=runtime.config.keybindings,
    )


def _pick_resume_session(manager: SessionManager, app: object) -> str:
    width = content_width(app.get_terminal_size(fallback=(80, 24)).columns)

    def render(previews: Sequence[SessionPreview]) -> None:
        print(resume_picker_line("recent zeta sessions:", width))
        for index, preview in enumerate(previews, start=1):
            print(resume_picker_line(format_picker_row(index, preview), width))

    return pick_resume_session(
        manager,
        render=render,
        prompt=resume_picker_line("select a session:", width - 1) + " ",
    )


def _apply_startup_theme(name: str | None, home: Path) -> tuple[str, ...]:
    """Apply the theme selected via settings; return dim notices for failures."""

    if name is None:
        _theme.set_active_palette(_theme.DARK)
        return ()
    palette, notice = _theme.resolve_palette(name, home=home)
    if palette is None:
        fallback = _theme.DARK
        _theme.set_active_palette(fallback)
        if notice is not None:
            return (notice,)
        return (f"theme · unknown theme {name!r}; using {fallback.name!r}",)
    _theme.set_active_palette(palette)
    return () if notice is None else (notice,)


def _validate_keybindings(remap: object) -> None:
    """Reject unknown actions and unparseable keys before opening the TUI."""

    try:
        resolve_keybindings(remap)  # type: ignore[arg-type]
    except KeybindingError as exc:
        raise SessionError(str(exc)) from exc


__all__ = ["RECENT_SESSION_LIMIT", "create_app", "format_picker_row"]
