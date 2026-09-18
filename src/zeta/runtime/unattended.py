"""Restricted composition for sessions without a human approval interface."""

from pathlib import Path

from ..core.approval import ApprovalDecision, ApprovalPolicy
from ..core.project_context import discover_repo_root
from ..core.session import OpenedSession, SessionManager
from ..loop import AgentLoop
from ..providers.factory import build_backend
from ..settings import load_settings
from ..skills import SkillCatalog
from ..skills.agent_catalog import discover_packaged_agents
from ..tools import ToolRegistry
from ..types import CompletionBackend


def build_unattended_loop(
    session: OpenedSession,
    *,
    home: Path,
    allow: tuple[str, ...],
    backend: CompletionBackend | None = None,
    router_mode: bool | None = None,
    router_style: str | None = None,
) -> AgentLoop:
    metadata, store = session.metadata, session.store
    if backend is None:
        backend, _model = build_backend(metadata.provider, metadata.model, home=home)
    policy = ApprovalPolicy(
        store=store, default=ApprovalDecision.DENY, always_allow=allow
    )
    if session.metadata.skill_catalog is None:
        raise ValueError("unattended sessions require a skill catalog")
    project_dir = discover_repo_root(Path(metadata.cwd)) / ".zeta"
    loaded = load_settings(home=home, project_dir=project_dir)
    if router_mode is None:
        router_mode = (
            True if loaded.settings.router is None else loaded.settings.router
        )
    if router_style is None:
        router_style = loaded.settings.router_style or "auto"
    jev_compaction = (
        True
        if loaded.settings.jev_compaction is None
        else loaded.settings.jev_compaction
    )
    skill_catalog = SkillCatalog.from_snapshot(session.metadata.skill_catalog)
    # Automation sessions never mount user-defined agents, even if metadata
    # was modified outside the restricted runner.
    agent_catalog = discover_packaged_agents()
    registry = ToolRegistry(
        metadata.cwd,
        session_store=store,
        approval_store=store,
        approval_policy=policy,
        enforce_approvals=True,
        skill_catalog=skill_catalog,
        agent_catalog=agent_catalog,
    )
    return AgentLoop(
        backend,
        store,
        skill_catalog=skill_catalog,
        agent_catalog=agent_catalog,
        registry=registry,
        approval_policy=policy,
        max_turns=25,
        skip_mcp_mount=True,
        system_prompt=metadata.system_prompt,
        router_mode=router_mode,
        router_style=router_style,
        jev_compaction=jev_compaction,
        on_completion_success=lambda: SessionManager(home).touch(metadata),
    )
