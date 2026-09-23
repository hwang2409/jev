import os


from contextlib import contextmanager


from pathlib import Path


import pytest


import zeta.tools._shared.sandbox as sandbox_module


import zeta.tools.read as read_module


from zeta.skills import SkillCatalog


from zeta.tools import ToolRegistry


from zeta.protocol.types import ToolCall


def test_path_from_fd_returns_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_text("content", encoding="utf-8")
    file_descriptor = os.open(target, os.O_RDONLY)
    try:
        assert sandbox_module._path_from_fd(file_descriptor) == str(target)
    finally:
        os.close(file_descriptor)


def test_ancestor_symlink_is_rejected(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("outside", encoding="utf-8")
    (sandbox / "dir").symlink_to(outside, target_is_directory=True)

    with (
        pytest.raises(ValueError, match="escaped session cwd"),
        sandbox_module.open_target(
            ToolRegistry(sandbox, skill_catalog=SkillCatalog.empty()),
            "dir/file",
            flags=os.O_RDONLY | os.O_CLOEXEC,
        ),
    ):
        pass


@pytest.mark.asyncio
async def test_post_walk_ancestry_check_rejects_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox = tmp_path / "sandbox"
    inner = sandbox / "inner"
    inner.mkdir(parents=True)
    (inner / "target").write_text("inside", encoding="utf-8")
    moved = tmp_path / "inner-moved"
    registry = ToolRegistry(sandbox, skill_catalog=SkillCatalog.empty())
    real_verify = sandbox_module._verify_ancestry
    calls = 0

    def verify(
        file_descriptor: int,
        identity: tuple[int, int],
        policy: sandbox_module.SandboxPolicy,
    ) -> None:
        nonlocal calls
        real_verify(file_descriptor, identity, policy)
        calls += 1
        if calls == 2:
            os.rename(inner, moved)

    monkeypatch.setattr(sandbox_module, "_verify_ancestry", verify)
    result = await registry.execute(
        ToolCall("rename-escape", "write", {"path": "inner/target", "content": "changed"})
    )

    assert result["isError"] is True
    assert "parent directory does not exist" in result["content"][0]["text"]
    assert (moved / "target").read_text(encoding="utf-8") == "inside"


def test_expand_user_path_expands_home() -> None:
    expanded = sandbox_module.expand_user_path("~/notes.txt")
    home = os.path.expanduser("~")
    assert expanded.startswith((home + os.sep, home))
    assert not expanded.startswith("~")


def test_expand_user_path_leaves_plain_paths_alone() -> None:
    assert sandbox_module.expand_user_path("relative/path") == "relative/path"
    assert sandbox_module.expand_user_path("/absolute/path") == "/absolute/path"


def test_expand_user_path_rejects_unknown_user() -> None:
    with pytest.raises(ValueError, match="use an absolute path"):
        sandbox_module.expand_user_path("~definitelynotarealuser-zeta68/x")


def test_sandbox_policy_resolves_tilde_and_classifies_roots(tmp_path: Path) -> None:
    policy = sandbox_module.SandboxPolicy(tmp_path)
    inside = policy.resolve("nested/file.txt")
    assert inside.in_cwd is True
    assert inside.absolute == tmp_path / "nested" / "file.txt"

    home_relative = policy.resolve("~/anywhere.txt")
    assert home_relative.absolute == Path(os.path.expanduser("~/anywhere.txt"))
    # Home is (almost always) outside the sandbox in tests.
    assert home_relative.in_cwd is False

    outside = policy.resolve(str(tmp_path.parent / "sibling.txt"))
    assert outside.in_cwd is False
    assert outside.absolute == tmp_path.parent / "sibling.txt"


def test_sandbox_policy_describe_roots_names_cwd(tmp_path: Path) -> None:
    policy = sandbox_module.SandboxPolicy(tmp_path)
    described = policy.describe_roots()
    assert str(tmp_path) in described
    assert "session cwd" in described
