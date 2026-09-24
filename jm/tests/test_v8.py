from __future__ import annotations

import io
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from jm.answers import JudgeResponse, NoulAnswer
from jm.cache import CacheStore
from jm.cli import main
from jm.presets import (
    PresetValidationError,
    load_preset,
    resolve_preset,
    validate_preset,
)

ROOT = Path(__file__).parents[1]


def _write_preset(tmp_path: Path, data: dict[str, object], name: str) -> Path:
    path = tmp_path / f"{name}.yml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_preset_defaults_are_resolved_and_stored(tmp_path: Path) -> None:
    data = deepcopy(dict(resolve_preset("jgrep").data))
    data["name"] = "small"
    data["chunking"].pop("limits")
    data.pop("compatible_chunkers")
    data["output"].pop("fields")
    data["output"].pop("pretty_template")
    path = _write_preset(tmp_path, data, "small")
    store = CacheStore(tmp_path / "cache")

    code = main(
        ["run", "--preset", str(path), "--query", "launch"],
        stdin=io.StringIO("launch decision\n"),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        judge_fn=lambda *_: JudgeResponse({"matches_query": NoulAnswer(0.9)}),
        cache_store=store,
    )

    assert code == 0
    entry = next(store.entries())
    battery = store.batteries.get("small", "1", entry.battery_hash)
    assert battery is not None
    assert battery.effective["chunking"]["limits"] == {
        "focus_bytes": 16_384,
        "context_field_bytes": 4_096,
        "state_bytes": 32_768,
    }
    assert battery.effective["compatible_chunkers"] == ["para"]
    assert battery.effective["output"]["pretty_template"] == (
        "{state_ref}\\t{answers.matches_query.noul}"
    )


def test_cwd_shadow_warns_but_explicit_path_does_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shadow = tmp_path / "jgrep.yml"
    shadow.write_text(
        (ROOT / "jm" / "presets" / "jgrep.yml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    preset = resolve_preset("jgrep")
    assert [item.code for item in preset.diagnostics] == ["preset_shadow"]
    assert load_preset(shadow).diagnostics == ()

    stderr = io.StringIO()
    assert (
        main(
            ["jgrep", "--query", "launch"],
            stdin=io.StringIO("launch\n"),
            stdout=io.StringIO(),
            stderr=stderr,
            judge_fn=lambda *_: JudgeResponse({"matches_query": NoulAnswer(0.9)}),
            cache_store=CacheStore(tmp_path / "cache"),
        )
        == 0
    )
    assert stderr.getvalue().count("preset_shadow") == 1

    explicit_stderr = io.StringIO()
    assert (
        main(
            ["run", "--preset", "./jgrep.yml", "--query", "launch"],
            stdin=io.StringIO("launch\n"),
            stdout=io.StringIO(),
            stderr=explicit_stderr,
            judge_fn=lambda *_: JudgeResponse({"matches_query": NoulAnswer(0.9)}),
            cache_store=CacheStore(tmp_path / "explicit-cache"),
        )
        == 0
    )
    assert "preset_shadow" not in explicit_stderr.getvalue()


def test_preset_show_is_yaml_and_version_is_available() -> None:
    stdout = io.StringIO()
    assert main(["preset", "show", "jgrep"], stdout=stdout, stderr=io.StringIO()) == 0
    assert yaml.safe_load(stdout.getvalue())["name"] == "jgrep"
    assert not stdout.getvalue().lstrip().startswith("{")


def test_unknown_wire_fields_warn_once_and_closed_blocks_fail(tmp_path: Path) -> None:
    data = deepcopy(dict(resolve_preset("jgrep").data))
    data["questions"]["matches_query"]["wire_hint"] = {"mode": "strict"}
    data["questions"]["matches_query"]["criteria"]["true"]["wire_note"] = "x"
    path = _write_preset(tmp_path, data, "wire-fields")
    loaded = load_preset(path)
    assert [item.path for item in loaded.diagnostics] == [
        "questions.matches_query.wire_hint",
        "questions.matches_query.criteria.true.wire_note",
    ]
    closed = deepcopy(data)
    closed["chunking"]["unknown"] = True
    with pytest.raises(PresetValidationError, match="chunking"):
        validate_preset(closed)


def test_v3_score_schema_accepts_three_levels_only() -> None:
    data = deepcopy(dict(resolve_preset("diff-risk-heat").data))
    data["schema"] = "jm.preset/v3"
    data["parameters"] = {"declared": []}
    data["questions"]["change_scope"]["criteria"] = data["questions"][
        "change_scope"
    ]["criteria"][:3]
    validate_preset(data)
    data["questions"]["change_scope"]["criteria"] = data["questions"][
        "change_scope"
    ]["criteria"][:2]
    with pytest.raises(PresetValidationError, match="three or four"):
        validate_preset(data)
