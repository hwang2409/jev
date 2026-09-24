"""Framework-neutral session startup shared by all frontends."""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from ..config.settings import LoadedSettings, ResolvedConfig, load_settings
from ..config.settings import resolve as resolve_settings
from ..core.approval import ApprovalPolicy
from ..core.project_context import (
    ProjectContext,
    PromptArgumentError,
    discover_repo_root,
    load_project_context,
    resolve_prompt_argument,
)
from ..core.session import SessionError, SessionManager, SessionMetadata, env_home
from ..protocol.types import CompletionBackend
from ..runtime.backend import build_backend
from ..runtime.cleanup import close_session
from ..runtime.composition import RuntimeComposition, compose_runtime
from ..runtime.loop import AgentLoop
from ..skills import SkillCatalog, discover_session_skills, replace_skill_index
from ..skills.agent_catalog import AgentCatalog, discover_session_agents

SettingsLoader = Callable[..., LoadedSettings]
ContextLoader = Callable[..., ProjectContext]
BackendBuilder = Callable[..., tuple[CompletionBackend, str]]
ResumePicker = Callable[[SessionManager], str]


@dataclass(slots=True)
class RuntimeBootstrap:
    """Common session state returned to a frontend wrapper."""

    composition: RuntimeComposition
    manager: SessionManager
    loaded_settings: LoadedSettings
    config: ResolvedConfig
    project_context: ProjectContext
    provider: str
    metadata: SessionMetadata
    resuming: bool
    override_on_resume: bool
    on_model_change: Callable[[str], None]


@dataclass(slots=True)
class HeadlessApp:
    """Session state needed by the headless driver."""

    loop: AgentLoop
    approval_policy: ApprovalPolicy
    ephemeral_root: Path | None

    async def close(self) -> None:
        await close_session(self.loop)


def create_runtime_bootstrap(
    args: argparse.Namespace,
    *,
    home: Path,
    ephemeral_root: Path | None,
    cleanup: ExitStack | None = None,
    resume_picker: ResumePicker | None = None,
    load_settings_fn: SettingsLoader = load_settings,
    load_project_context_fn: ContextLoader = load_project_context,
    backend_builder: BackendBuilder = build_backend,
    persist_plan_mode: bool = False,
) -> RuntimeBootstrap:
    """Create one session for either the TUI or a non-interactive frontend."""

    manager = SessionManager(ephemeral_root if ephemeral_root is not None else home)
    try:
        system_override = resolve_prompt_argument(getattr(args, "system_prompt", None))
        system_append = resolve_prompt_argument(
            getattr(args, "append_system_prompt", None)
        )
    except PromptArgumentError as exc:
        raise SessionError(str(exc)) from exc

    repo_root = discover_repo_root(Path.cwd())
    loaded_settings = load_settings_fn(home=home, project_dir=repo_root / ".zeta")
    config = resolve_settings(
        loaded_settings.settings,
        cli_provider=getattr(args, "provider", None),
        cli_model=getattr(args, "model", None),
        cli_router=getattr(args, "router", None),
        cli_router_style=getattr(args, "router_style", None),
        cli_jev_compaction=getattr(args, "jev_compaction", None),
        cli_memory_injection=getattr(args, "memory_injection", None),
        cli_yolo=getattr(args, "yolo", None),
        cli_safety_tier=getattr(args, "safety_tier", None),
        cli_token_budget=getattr(args, "token_budget", None),
        cli_memory_config=getattr(args, "memory_config", None),
    )

    resume_id = getattr(args, "resume", None)
    continue_session = bool(getattr(args, "continue_session", False))
    force_provider = bool(getattr(args, "force_provider", False))
    resuming = continue_session or resume_id is not None
    if force_provider and not resuming:
        raise SessionError("--force-provider requires --continue or --resume")
    if force_provider and config.model is None:
        raise SessionError("--force-provider requires --model")

    opened = None
    pending_override: tuple[str | None, str | None] | None = None
    if resuming:
        if resume_id == "":
            if resume_picker is None:
                resume_id = _pick_resume_session(manager)
            else:
                resume_id = resume_picker(manager)
        opened = (
            manager.open(resume_id)
            if resume_id is not None
            else manager.open(manager.find_most_recent(cwd=Path.cwd()).session_id)
        )
        if cleanup is not None:
            cleanup.enter_context(opened.store)
        metadata = opened.metadata
        skill_catalog = _session_skill_catalog(metadata, home, manager)
        agent_catalog = _session_agent_catalog(metadata, home, manager)
        cli_provider = getattr(args, "provider", None)
        cli_model = getattr(args, "model", None)
        provider_override = cli_provider or loaded_settings.settings.provider
        model_override = cli_model or loaded_settings.settings.model
        mismatches = []
        if provider_override is not None and provider_override != metadata.provider:
            mismatches.append(
                f"provider {provider_override!r} does not match {metadata.provider!r}"
            )
        if model_override is not None and model_override != metadata.model:
            mismatches.append(
                f"model {model_override!r} does not match {metadata.model!r}"
            )
        if mismatches and not force_provider:
            raise SessionError(
                f"session override rejected: {'; '.join(mismatches)}; "
                "use --force-provider to override"
            )
        provider = provider_override or metadata.provider
        model = model_override or metadata.model
        override_on_resume = system_override is not None or system_append is not None
        if metadata.system_prompt and not override_on_resume:
            project_context = ProjectContext(
                metadata.system_prompt,
                tuple(Path(path) for path in metadata.context_files),
            )
        else:
            project_context = load_project_context_fn(
                cwd=Path(metadata.cwd),
                repo_root=discover_repo_root(Path(metadata.cwd)),
                zeta_home=home,
                system_override=system_override,
                system_append=system_append,
                catalog=skill_catalog,
            )
            persisted = manager.persist_context_snapshot(
                metadata,
                system_prompt=project_context.system_prompt,
                context_files=[str(path) for path in project_context.files],
                overwrite=override_on_resume,
            )
            project_context = ProjectContext(
                persisted.system_prompt,
                tuple(Path(path) for path in persisted.context_files),
                project_context.notices,
            )
        if mismatches and force_provider:
            pending_override = (
                provider if provider != metadata.provider else None,
                model if model != metadata.model else None,
            )
    else:
        provider = config.provider
        model = config.model
        skill_catalog = discover_session_skills(home=home, project_dir=repo_root)
        agent_catalog = discover_session_agents(home=home, project_dir=repo_root)
        project_context = load_project_context_fn(
            cwd=Path.cwd(),
            repo_root=repo_root,
            zeta_home=home,
            system_override=system_override,
            system_append=system_append,
            catalog=skill_catalog,
        )
        override_on_resume = False

    def completion_success() -> None:
        nonlocal pending_override
        if pending_override is not None:
            manager.record_override(
                metadata,
                provider=pending_override[0],
                model=pending_override[1],
            )
            pending_override = None
            return
        manager.touch(metadata)

    def model_changed(model_name: str) -> None:
        nonlocal pending_override
        if pending_override is not None:
            pending_override = (provider, model_name)
            return
        manager.record_override(metadata, provider=None, model=model_name)

    plan_mode_callback = None
    if persist_plan_mode:
        plan_mode_callback = lambda enabled: manager.record_plan_mode(
            metadata, enabled=enabled
        )

    composition = compose_runtime(
        home=home,
        cwd=Path.cwd(),
        manager=manager,
        config=config,
        provider=provider,
        model=model,
        project_context=project_context,
        backend_builder=backend_builder,
        opened=opened,
        on_completion_success=completion_success,
        on_plan_mode_change=plan_mode_callback,
        max_turns=getattr(args, "max_turns", None),
        skill_catalog=skill_catalog,
        agent_catalog=agent_catalog,
    )
    if opened is None:
        opened = composition.opened
        if cleanup is not None:
            cleanup.enter_context(opened.store)
    metadata = opened.metadata
    return RuntimeBootstrap(
        composition=composition,
        manager=manager,
        loaded_settings=loaded_settings,
        config=config,
        project_context=project_context,
        provider=provider,
        metadata=metadata,
        resuming=resuming,
        override_on_resume=override_on_resume,
        on_model_change=model_changed,
    )


def _pick_resume_session(manager: SessionManager) -> str:
    previews = manager.list_session_previews(limit=20)
    if not previews:
        raise SessionError("no prior zeta session found")
    try:
        selected = int(input("select a session: ").strip())
        if not 1 <= selected <= len(previews):
            raise ValueError("selection out of range")
        return previews[selected - 1].session_id
    except (EOFError, ValueError, IndexError) as exc:
        raise SessionError("invalid resume session selection") from exc


def _session_skill_catalog(
    metadata: SessionMetadata, home: Path, manager: SessionManager
) -> SkillCatalog:
    if metadata.skill_catalog is None:
        catalog = discover_session_skills(
            home=home, project_dir=discover_repo_root(Path(metadata.cwd))
        )
        try:
            persisted = manager.persist_skill_catalog(
                metadata,
                catalog,
                system_prompt=(
                    replace_skill_index(metadata.system_prompt, catalog)
                    if metadata.system_prompt
                    else None
                ),
            )
            return SkillCatalog.from_snapshot(persisted.skill_catalog)
        except ValueError as exc:
            raise SessionError("session skill catalog is invalid") from exc
    try:
        return SkillCatalog.from_snapshot(metadata.skill_catalog)
    except ValueError as exc:
        raise SessionError("session skill catalog is invalid") from exc


def _session_agent_catalog(
    metadata: SessionMetadata, home: Path, manager: SessionManager
) -> AgentCatalog:
    if metadata.agent_catalog is None:
        catalog = discover_session_agents(
            home=home, project_dir=discover_repo_root(Path(metadata.cwd))
        )
        persisted = manager.persist_agent_catalog(metadata, catalog)
        try:
            return AgentCatalog.from_snapshot(persisted.agent_catalog)
        except ValueError as exc:
            raise SessionError("session agent catalog is invalid") from exc
    try:
        return AgentCatalog.from_snapshot(metadata.agent_catalog)
    except ValueError as exc:
        raise SessionError("session agent catalog is invalid") from exc


def create_headless_app(args: argparse.Namespace) -> HeadlessApp:
    """Create a session without constructing frontend-specific objects."""

    home = env_home()
    ephemeral_root = (
        Path(tempfile.mkdtemp(prefix="zeta-ephemeral-"))
        if bool(getattr(args, "no_session", False))
        else None
    )
    try:
        runtime = create_runtime_bootstrap(
            args,
            home=home,
            ephemeral_root=ephemeral_root,
        )
        return HeadlessApp(
            runtime.composition.loop,
            runtime.composition.policy,
            ephemeral_root,
        )
    except BaseException:
        if ephemeral_root is not None:
            import shutil

            shutil.rmtree(ephemeral_root, ignore_errors=True)
        raise


__all__ = [
    "HeadlessApp",
    "RuntimeBootstrap",
    "create_headless_app",
    "create_runtime_bootstrap",
]
