from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from jm.presets import (
    PresetNotFoundError,
    PresetUsageError,
    PresetValidationError,
    load_preset,
    resolve_chunker,
    resolve_preset,
    validate_preset,
)

ROOT = Path(__file__).parents[1]
PRESETS = ROOT / "jm" / "presets"
HISTORICAL_TOOL_NAME = "j" + "map"
PLAN = (
    ROOT
    / "docs"
    / "superpowers"
    / "plans"
    / f"2026-09-22-{HISTORICAL_TOOL_NAME}-v1.md"
)
BUILTINS = ("jgrep", "jfilter", "diff-risk-heat")


def test_builtins_have_expected_metadata_and_batteries() -> None:
    expected = {
        "jgrep": ("para", ("line", "para", "file"), 1, {"noul"}),
        "jfilter": ("record", ("record",), 1, {"noul"}),
        "diff-risk-heat": ("hunk", ("hunk",), 9, {"noul", "score"}),
    }
    for name in BUILTINS:
        preset = resolve_preset(name)
        assert (
            preset.default_chunker,
            preset.compatible_chunkers,
            len(preset.questions),
            {question["type"] for question in preset.questions.values()},
        ) == expected[name]
        assert preset.version == "1"
        assert preset.model == "typesafe-ai/jev"
        assert preset.chunking["limits"] == {
            "focus_bytes": 16384,
            "context_field_bytes": 4096,
            "state_bytes": 32768,
        }
        assert preset.data["output"]["fields"] == [
            "record_type",
            "state_ref",
            "source_ref",
            "answers",
            "error",
            "missing_questions",
            "coverage",
            "coverage_counts",
            "coverage_reasons",
            "meta",
        ]


def test_builtins_match_the_plan_blocks_byte_for_byte() -> None:
    plan = PLAN.read_text(encoding="utf-8")
    for name in BUILTINS:
        marker = (
            f"`{HISTORICAL_TOOL_NAME}/{HISTORICAL_TOOL_NAME}/presets/"
            f"{name}.yml`:\n\n```yaml\n"
        )
        block = (
            plan.split(marker, 1)[1]
            .split("\n```", 1)[0]
            .replace(HISTORICAL_TOOL_NAME, "jm")
            + "\n"
        )
        assert (PRESETS / f"{name}.yml").read_text(encoding="utf-8") == block


@pytest.mark.parametrize("missing", [
    "schema",
    "name",
    "version",
    "model",
    "chunking",
    "compatible_chunkers",
    "questions",
    "thresholds",
    "output",
])
def test_validation_requires_every_top_level_field(missing: str) -> None:
    data = yaml.safe_load((PRESETS / "jgrep.yml").read_text(encoding="utf-8"))
    del data[missing]
    with pytest.raises(PresetValidationError, match="required"):
        validate_preset(data)


def test_validation_rejects_unknown_fields_and_alias_models() -> None:
    data = yaml.safe_load((PRESETS / "jgrep.yml").read_text(encoding="utf-8"))
    data["unexpected"] = True
    with pytest.raises(PresetValidationError, match="unknown"):
        validate_preset(data)

    data = copy.deepcopy(data)
    data.pop("unexpected")
    data["model"] = "jev-latest"
    with pytest.raises(PresetValidationError, match="model"):
        validate_preset(data)


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda data: data["questions"]["matches_query"].update(criteria={}),
            "true and false",
        ),
        (
            lambda data: data["questions"]["matches_query"].update(type="other"),
            "one of",
        ),
        (lambda data: data["compatible_chunkers"].append("other"), "unknown"),
        (
            lambda data: data["thresholds"]["matches_query"].update(
                fail_at_least=0.5
            ),
            "exactly one",
        ),
        (
            lambda data: data["thresholds"]["matches_query"].update(
                keep_at_least=2
            ),
            "0 to 1",
        ),
    ],
)
def test_validation_rejects_invalid_question_or_threshold_data(change, message) -> None:
    data = yaml.safe_load((PRESETS / "jgrep.yml").read_text(encoding="utf-8"))
    change(data)
    with pytest.raises(PresetValidationError, match=message):
        validate_preset(data)


def test_load_rejects_reserved_policy_keyword_question_id(tmp_path: Path) -> None:
    data = yaml.safe_load((PRESETS / "jgrep.yml").read_text(encoding="utf-8"))
    data["questions"]["any"] = data["questions"].pop("matches_query")
    data["thresholds"]["any"] = data["thresholds"].pop("matches_query")
    path = tmp_path / "reserved.yml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(PresetValidationError, match="reserved.*policy keyword"):
        load_preset(path)


def test_lookup_order_is_explicit_then_cwd_then_builtin_then_user(
    tmp_path, monkeypatch
) -> None:
    content = (PRESETS / "jgrep.yml").read_text(encoding="utf-8")
    cwd = tmp_path / "cwd"
    builtins = tmp_path / "builtins"
    user = tmp_path / "user"
    for directory in (cwd, builtins, user):
        directory.mkdir()
        (directory / "jgrep.yml").write_text(content, encoding="utf-8")
    explicit = tmp_path / "explicit.yml"
    explicit.write_text(content, encoding="utf-8")
    monkeypatch.setenv("JM_PRESETS", str(user))

    assert resolve_preset("jgrep", explicit_path=explicit).path == explicit.resolve()
    assert resolve_preset("jgrep", cwd=cwd, package_dir=builtins).path == (
        cwd / "jgrep.yml"
    ).resolve()
    (cwd / "jgrep.yml").unlink()
    assert resolve_preset("jgrep", cwd=cwd, package_dir=builtins).path == (
        builtins / "jgrep.yml"
    ).resolve()
    (builtins / "jgrep.yml").unlink()
    assert resolve_preset("jgrep", cwd=cwd, package_dir=builtins).path == (
        user / "jgrep.yml"
    ).resolve()


def test_incompatible_chunker_is_usage_error_with_allowed_set() -> None:
    preset = resolve_preset("jgrep")
    with pytest.raises(PresetUsageError, match="line, para, file") as error:
        resolve_chunker(preset, "record")
    assert error.value.exit_code == 64
    assert resolve_chunker(preset) == "para"
    assert resolve_chunker(preset, "file") == "file"


@pytest.mark.parametrize(
    "name",
    ["../x", "a/b", "/tmp/jgrep.yml"],
)
def test_lookup_name_rejects_paths(name: str, tmp_path: Path) -> None:
    with pytest.raises(PresetNotFoundError, match="safe preset name"):
        resolve_preset(name, cwd=tmp_path, package_dir=tmp_path, user_dir=tmp_path)


@pytest.mark.parametrize("name", BUILTINS)
def test_each_preset_rejects_incompatible_by(name: str) -> None:
    preset = resolve_preset(name)
    incompatible = next(
        chunker for chunker in ("line", "para", "hunk", "file", "record")
        if chunker not in preset.compatible_chunkers
    )
    with pytest.raises(PresetUsageError, match="incompatible"):
        resolve_chunker(preset, incompatible)


def test_malformed_preset_has_clear_error_without_traceback(tmp_path: Path) -> None:
    path = tmp_path / "malformed.yml"
    path.write_text("schema: [", encoding="utf-8")

    with pytest.raises(PresetValidationError) as error:
        load_preset(path)

    message = str(error.value)
    assert "invalid YAML" in message
    assert "Traceback" not in message


def test_duplicate_yaml_keys_are_rejected(tmp_path) -> None:
    path = tmp_path / "duplicate.yml"
    path.write_text(
        """schema: jm.preset/v1\nschema: jm.preset/v1\n""",
        encoding="utf-8",
    )
    with pytest.raises(PresetValidationError, match="duplicate"):
        load_preset(path)
