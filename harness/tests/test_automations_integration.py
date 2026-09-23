from __future__ import annotations


from pathlib import Path


from zeta.core.approval import ApprovalDecision, ApprovalPolicy


from zeta.core.store import ConversationStore


from zeta.skills import SkillCatalog


from zeta.tools import ToolRegistry


from zeta.tools.agent import ChildApprovalPolicy


from zeta.protocol.types import ToolCall


import asyncio


import json


from dataclasses import replace


from datetime import UTC, datetime, timedelta


import pytest


from zeta.automations import commands


from zeta.automations.authoring import import_jobs, listing, resolve_job


from zeta.automations.daemon import daemon_lock, serve


from zeta.automations.delivery import SlackDelivery


from zeta.automations.models import instant, parse_job


from zeta.automations.runner import poll_events, run_claimed


from zeta.automations.services import validate_permissions


from zeta.automations.store import SQLiteStore


from zeta.automations.tick import tick


from zeta.automations.trigger import Schedule, cron_matches, parse_trigger


from zeta.core.fake import FakeBackend, ScriptedTurn


from zeta.core.session import SessionManager


from zeta.core.slash import create_slash_registry


from zeta.mcp.config import load_mcp_config, server_to_json


from zeta.mcp.mount import MCPMount


from zeta.prompts import load_identity


from zeta.runtime.unattended import build_unattended_loop


from zeta.skills import discover_session_skills


from zeta.protocol.types import TextContent


START = datetime(2026, 9, 9, 11, 0, tzinfo=UTC)


DUE = START + timedelta(hours=1)


def _job(tmp_path: Path, name: str = "brief", *, poll: bool = False):
    return parse_job(
        name,
        {
            "prompt": "Summarize new activity.",
            "trigger": {"kind": "poll", "condition": "New urgent activity"}
            if poll
            else {"kind": "schedule", "cron": "0 8 * * 1-5"},
            "servers": ["slack"],
            "allow": [],
            "deliver": "slack:U123",
            "provider": "fake",
            "model": "fake",
            "cwd": str(tmp_path),
        },
    )


def _arm(store: SQLiteStore, job, now: datetime = START) -> None:
    state = store.draft(job)
    store.approve(job.name, state.revision, "U123", now)


class RecordingDelivery:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, str, str]] = []

    async def resolve(self, target: str) -> str:
        return "U123"

    async def send(self, recipient: str, name: str, session_id: str, text: str) -> str:
        self.sent.append((recipient, name, session_id, text))
        return '{"ts":"123.456"}'


async def _empty_mount(job, registry: ToolRegistry, home: Path) -> MCPMount:
    return MCPMount(registry, {})


class SlackClient:
    def __init__(self, search_text: str = "User U123") -> None:
        self.calls = []
        self.search_text = search_text

    async def call_tool(self, name, arguments, abort_signal):
        self.calls.append((name, arguments))
        return {
            "content": [
                {
                    "type": "text",
                    "text": self.search_text if "search" in name else '{"ts":"123"}',
                }
            ],
            "isError": False,
        }


class SlackMount:
    def __init__(self, client) -> None:
        self.client = client

    def client_for(self, name):
        return self.client


async def test_unattended_allow_list_gates_even_exempt_and_internal_calls(
    tmp_path: Path,
) -> None:
    policy = ApprovalPolicy(
        default=ApprovalDecision.DENY, always_allow=["permitted(ok*)"]
    )
    store = ConversationStore(tmp_path)
    registry = ToolRegistry(
        tmp_path,
        register_builtin=False,
        approval_policy=policy,
        approval_store=store,
        enforce_approvals=True,
        skill_catalog=SkillCatalog.empty(),
    )
    calls = []
    for name in ("permitted", "forbidden"):
        registry.register(
            name,
            lambda args: calls.append(args) or "executed",
            parameters={"type": "object", "properties": {"value": {"type": "string"}}},
            approval_subject="value",
            requires_approval=False,
        )
    allowed = await registry.execute(ToolCall("1", "permitted", {"value": "okay"}))
    denied = await registry.execute(
        ToolCall("2", "forbidden", {"value": "okay"}), _skip_approval=True
    )
    scoped = await registry.execute(ToolCall("3", "permitted", {"value": "bad"}))
    assert not allowed["isError"]
    assert denied["isError"] and scoped["isError"]
    assert calls == [{"value": "okay"}]
    child_store = ConversationStore(tmp_path / "child")
    child = registry.clone_for_session(child_store)
    child.set_approval_policy(
        ChildApprovalPolicy(policy, child_store, "child", "child-1")
    )
    denied_child = await child.execute(ToolCall("4", "forbidden", {"value": "okay"}))
    assert denied_child["isError"]
    assert registry.denied_tools == ["forbidden", "permitted", "forbidden"]
    assert calls == [{"value": "okay"}]
    await child.close()
    await registry.close()


async def test_automation_catalog_excludes_project_skills_from_prompt_and_tool(
    tmp_path: Path,
) -> None:
    project_skill = tmp_path / ".zeta" / "skills" / "project.md"
    project_skill.parent.mkdir(parents=True)
    project_skill.write_text(
        "---\nname: project\ndescription: project skill\n---\n\nproject body\n",
        encoding="utf-8",
    )
    catalog = discover_session_skills(home=tmp_path)
    session = SessionManager(tmp_path).create(
        provider="fake",
        model="fake",
        cwd=tmp_path,
        system_prompt=load_identity(catalog=catalog),
        skill_catalog=catalog,
    )
    loop = build_unattended_loop(
        session,
        home=tmp_path,
        allow=("skill",),
        backend=FakeBackend([]),
    )
    try:
        loaded = await loop.tool_registry.execute(
            ToolCall("project", "skill", {"name": "project"})
        )
        assert "project skill" not in session.metadata.system_prompt
        assert [item["name"] for item in session.metadata.skill_catalog or []] == [
            "review"
        ]
        assert loaded["isError"] is True
        assert "available skills: review" in loaded["content"][0]["text"]
    finally:
        await loop.close()
