from pathlib import Path


import pytest


from zeta.skills.loader import (
    SkillCatalog,
    SkillMeta,
    discover_session_skills,
    discover_skills,
    load_skill,
)


from zeta.tools import ToolRegistry


from zeta.protocol.types import ToolCall


def _write_skill(path: Path, name: str, body: str, *, keywords: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {name} description\n"
        f"{keywords}"
        "---\n\n"
        f"{body}\n",
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_skill_tool_loads_and_reports_unknown_name(tmp_path: Path) -> None:
    from zeta.skills.loader import discover_packaged_skills

    registry = ToolRegistry(tmp_path, skill_catalog=discover_packaged_skills())

    loaded = await registry.execute(ToolCall("skill-load", "skill", {"name": "review"}))
    unknown = await registry.execute(
        ToolCall("skill-unknown", "skill", {"name": "missing"})
    )

    assert loaded["isError"] is False
    assert "Review the requested code change." in loaded["content"][0]["text"]
    assert unknown["isError"] is True
    assert "available skills: review" in unknown["content"][0]["text"]


@pytest.mark.asyncio
async def test_directory_skill_tool_reports_resource_directory(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "bundle"
    _write_skill(skill_dir / "SKILL.md", "bundle", "bundle body")
    catalog = discover_session_skills(home=tmp_path)
    registry = ToolRegistry(tmp_path, skill_catalog=catalog)

    loaded = await registry.execute(
        ToolCall("skill-bundle", "skill", {"name": "bundle"})
    )

    assert loaded["isError"] is False
    assert str(skill_dir.resolve()) in loaded["content"][0]["text"]
