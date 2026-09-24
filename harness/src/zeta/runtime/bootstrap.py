"""Framework-neutral session startup for non-interactive frontends."""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..config.settings import load_settings
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
from ..runtime.backend import build_backend
from ..runtime.cleanup import close_session
from ..runtime.composition import compose_runtime
from ..runtime.loop import AgentLoop
from ..skills import SkillCatalog, discover_session_skills
from ..skills.agent_catalog import AgentCatalog, discover_session_agents


@dataclass(slots=True)
class HeadlessApp:
    """Session state needed by the headless driver."""

    loop: AgentLoop
    approval_policy: ApprovalPolicy
    ephemeral_root: Path | None

    async def close(self) -> None:
        await close_session(self.loop)


def create_headless_app(args: argparse.Namespace) -> HeadlessApp:
    """Create a session without constructing frontend-specific objects."""

    home = env_home()
    ephemeral_root = (
        Path(tempfile.mkdtemp(prefix="zeta-ephemeral-"))
        if bool(getattr(args, "no_session", False))
        else None
    )
    try:
        return _create_headless_app(args, home, ephemeral_root)
    except BaseException:
        if ephemeral_root is not None:
            import shutil

            shutil.rmtree(ephemeral_root, ignore_errors=True)
        raise


def _create_headless_app(
    args: argparse.Namespace,
    home: Path,
    ephemeral_root: Path | None,
) -> HeadlessApp:
    manager = SessionManager(ephemeral_root if ephemeral_root is not None else home)
    try:
        system_override = resolve_prompt_argument(
            getattr(args, "system_prompt", None)
        )
        system_append = resolve_prompt_argument(
            getattr(args, "append_system_prompt", None)
        )
    except PromptArgumentError as exc:
        raise SessionError(str(exc)) from exc

    repo_root = discover_repo_root(Path.cwd())
    loaded_settings = load_settings(home=home, project_dir=repo_root / ".zeta")
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
    if force_provider and resume_id is None and not continue_session:
        raise SessionError("--force-provider requires --continue or --resume")
    if force_provider and config.model is None:
        raise SessionError("--force-provider requires --model")
    opened = None
    pending_override: tuple[str | None, str | None] | None = None
    if resume_id is not None or continue_session:
        if resume_id == "":
            previews = manager.list_session_previews(limit=20)
            if not previews:
                raise SessionError("no prior zeta session found")
            try:
                selected = int(input("select a session: ").strip())
                resume_id = previews[selected - 1].session_id
            except (EOFError, ValueError, IndexError) as exc:
                raise SessionError("invalid resume session selection") from exc
        if resume_id is not None:
            opened = manager.open(resume_id)
        else:
            recent = manager.find_most_recent(cwd=Path.cwd())
            opened = manager.open(recent.session_id)
        metadata = opened.metadata
        cli_provider = getattr(args, "provider", None)
        cli_model = getattr(args, "model", None)
        provider = cli_provider or metadata.provider
        model = cli_model or metadata.model
        mismatches = []
        if cli_provider is not None and cli_provider != metadata.provider:
            mismatches.append(
                f"provider {cli_provider!r} does not match {metadata.provider!r}"
            )
        if cli_model is not None and cli_model != metadata.model:
            mismatches.append(
                f"model {cli_model!r} does not match {metadata.model!r}"
            )
        if mismatches and not force_provider:
            raise SessionError(
                f"session override rejected: {'; '.join(mismatches)}; "
                "use --force-provider to override"
            )
        skill_catalog = _session_skill_catalog(metadata, home, manager)
        agent_catalog = _session_agent_catalog(metadata, home, manager)
        override_on_resume = system_override is not None or system_append is not None
        project_context = (
            ProjectContext(
                metadata.system_prompt,
                tuple(Path(path) for path in metadata.context_files),
            )
            if metadata.system_prompt and not override_on_resume
            else load_project_context(
                cwd=Path(metadata.cwd),
                repo_root=discover_repo_root(Path(metadata.cwd)),
                zeta_home=home,
                system_override=system_override,
                system_append=system_append,
                catalog=skill_catalog,
            )
        )
        if override_on_resume:
            persisted = manager.persist_context_snapshot(
                metadata,
                system_prompt=project_context.system_prompt,
                context_files=[str(path) for path in project_context.files],
                overwrite=True,
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
        skill_catalog = discover_session_skills(
            home=home, project_dir=repo_root / ".zeta"
        )
        agent_catalog = discover_session_agents(
            home=home, project_dir=repo_root / ".zeta"
        )
        project_context = load_project_context(
            cwd=Path.cwd(),
            repo_root=repo_root,
            zeta_home=home,
            system_override=system_override,
            system_append=system_append,
            catalog=skill_catalog,
        )

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

    composition = compose_runtime(
        home=home,
        cwd=Path.cwd(),
        manager=manager,
        config=config,
        provider=provider,
        model=model,
        project_context=project_context,
        backend_builder=build_backend,
        opened=opened,
        on_completion_success=completion_success,
        max_turns=getattr(args, "max_turns", None),
        skill_catalog=skill_catalog,
        agent_catalog=agent_catalog,
    )
    if opened is None:
        opened = composition.opened
    metadata = opened.metadata
    return HeadlessApp(composition.loop, composition.policy, ephemeral_root)


def _session_skill_catalog(
    metadata: SessionMetadata, home: Path, manager: SessionManager
) -> SkillCatalog:
    if metadata.skill_catalog is not None:
        return SkillCatalog.from_snapshot(metadata.skill_catalog)
    catalog = discover_session_skills(
        home=home, project_dir=discover_repo_root(Path(metadata.cwd))
    )
    persisted = manager.persist_skill_catalog(metadata, catalog)
    return SkillCatalog.from_snapshot(persisted.skill_catalog)


def _session_agent_catalog(
    metadata: SessionMetadata, home: Path, manager: SessionManager
) -> AgentCatalog:
    if metadata.agent_catalog is not None:
        return AgentCatalog.from_snapshot(metadata.agent_catalog)
    catalog = discover_session_agents(
        home=home, project_dir=discover_repo_root(Path(metadata.cwd))
    )
    persisted = manager.persist_agent_catalog(metadata, catalog)
    return AgentCatalog.from_snapshot(persisted.agent_catalog)


__all__ = ["HeadlessApp", "create_headless_app"]
