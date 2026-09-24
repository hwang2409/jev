from __future__ import annotations

import hashlib
import io
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from jm.answers import CoverageRecord, JudgeResponse, NoulAnswer, RecordMeta
from jm.cache import CacheStore, _parse_entry, battery_hash
from jm.cli import main
from jm.client import build_canonical_request
from jm.gates import compile_policy
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
    data["chunking"].pop("context_lines")
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


def test_omitted_boilerplate_matches_explicit_defaults() -> None:
    explicit = deepcopy(dict(resolve_preset("jfilter").data))
    explicit["compatible_chunkers"] = ["record"]
    omitted = deepcopy(explicit)
    omitted["chunking"].pop("limits")
    omitted.pop("compatible_chunkers")
    omitted["output"].pop("fields")
    omitted["output"].pop("pretty_template")

    resolved = validate_preset(omitted)
    assert resolved["chunking"] == explicit["chunking"]
    assert resolved["compatible_chunkers"] == explicit["compatible_chunkers"]
    assert resolved["output"] == explicit["output"]


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

    bare_stdout = io.StringIO()
    bare_stderr = io.StringIO()
    assert (
        main(
            ["preset", "validate", "jgrep.yml"],
            stdout=bare_stdout,
            stderr=bare_stderr,
        )
        == 0
    )
    assert bare_stderr.getvalue().count("preset_shadow") == 1

    for target in ("./jgrep.yml", str(shadow.resolve())):
        stdout = io.StringIO()
        stderr = io.StringIO()
        assert main(["preset", "validate", target], stdout=stdout, stderr=stderr) == 0
        assert "preset_shadow" not in stderr.getvalue()
        assert stdout.getvalue() == bare_stdout.getvalue()


def test_preset_show_is_yaml_and_version_is_available() -> None:
    stdout = io.StringIO()
    assert main(["preset", "show", "jgrep"], stdout=stdout, stderr=io.StringIO()) == 0
    shown = yaml.safe_load(stdout.getvalue())
    assert shown["name"] == "jgrep"
    assert shown == resolve_preset("jgrep").data
    assert not stdout.getvalue().lstrip().startswith("{")


def test_help_audits_v8_options_and_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as help_exit:
        main(["run", "--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    descriptions = {
        "--by": "form states by line, paragraph, hunk, file, record, or state",
        "--max-chunks": "cap formed states; excess states become scan-cap skips",
        "--format": "write canonical JSONL, or also render results as pretty text",
        "--filter": "select visible results with the preset threshold or a policy",
        "--query": "supply the preset's query parameter",
    }
    for option, description in descriptions.items():
        assert option in help_text
        assert description in normalized_help

    with pytest.raises(SystemExit) as version_exit:
        main(["--version"])
    assert version_exit.value.code == 0
    assert capsys.readouterr().out.startswith("jm ")


def test_line_context_pins_jgrep_and_keeps_generic_default(tmp_path: Path) -> None:
    input_text = "one\ntwo\nthree\n"
    jgrep_states = []
    assert main(
        ["run", "--preset", "jgrep", "--query", "x", "--by", "line"],
        stdin=io.StringIO(input_text),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        judge_fn=lambda state, *_: (
            jgrep_states.append(state)
            or JudgeResponse({"matches_query": NoulAnswer(0.9)})
        ),
        cache_store=CacheStore(tmp_path / "jgrep-cache"),
    ) == 0
    assert {
        state.state_ref: state.context["surrounding"] for state in jgrep_states
    } == {"stdin#L1": [], "stdin#L2": [], "stdin#L3": []}

    generic = deepcopy(dict(resolve_preset("jgrep").data))
    generic["name"] = "generic-line"
    generic["chunking"] = {
        "by": "line",
        "limits": generic["chunking"]["limits"],
    }
    generic["compatible_chunkers"] = ["line"]
    path = _write_preset(tmp_path, generic, "generic-line")
    generic_states = []
    assert main(
        ["run", "--preset", str(path), "--query", "x"],
        stdin=io.StringIO(input_text),
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        judge_fn=lambda state, *_: (
            generic_states.append(state)
            or JudgeResponse({"matches_query": NoulAnswer(0.9)})
        ),
        cache_store=CacheStore(tmp_path / "generic-cache"),
    ) == 0
    assert {
        state.state_ref: state.context["surrounding"] for state in generic_states
    } == {
        "stdin#L1": ["two"],
        "stdin#L2": ["one", "three"],
        "stdin#L3": ["two"],
    }


def test_builtin_yaml_bytes_and_hashes_are_stable() -> None:
    expected_hashes = {
        "jgrep": (
            "996a27c78125b7259e13a3ad0b096841234cc747d19874859c3b43385b6fa10e"
        ),
        "jfilter": (
            "cc18319473d99e07a5720833f6add483da1dfd6cc6f3c5e56ece9986860914c1"
        ),
        "diff-risk-heat": (
            "94b4ac588204e2caaec1baed2594bbf2031e94a2df08767b27273e9ac1f8f062"
        ),
    }
    expected_battery_hashes = {
        "jgrep": (
            "sha256:4cb9314fa183703efb2be9a4ee2dcb8693476de9d6532075fe9ffb73e696ef46"
        ),
        "jfilter": (
            "sha256:d5e7bbb4261533d4ea68db48f7a014e9edb6b9cf6786899d164a24dec3bede98"
        ),
        "diff-risk-heat": (
            "sha256:149c3b974902788ffe531a858c05eb746a61c673233ae8a39a58e3868e9fdddc"
        ),
    }
    plan = (
        ROOT / "docs" / "superpowers" / "plans" / "2026-09-22-jmap-v1.md"
    ).read_text(encoding="utf-8")
    for name, expected_hash in expected_hashes.items():
        path = ROOT / "jm" / "presets" / f"{name}.yml"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash
        marker = f"`jmap/jmap/presets/{name}.yml`:\n\n```yaml\n"
        block = (
            plan.split(marker, 1)[1].split("\n```", 1)[0].replace("jmap", "jm")
            + "\n"
        )
        assert path.read_text(encoding="utf-8") == block

        assert (
            battery_hash(resolve_preset(name).questions)
            == expected_battery_hashes[name]
        )


@pytest.mark.parametrize("context_lines", [0, 2])
def test_line_compatible_context_lines_is_allowed_and_para_only_is_closed(
    tmp_path: Path, context_lines: int
) -> None:
    compatible = deepcopy(dict(resolve_preset("jgrep").data))
    compatible["name"] = "line-compatible"
    compatible["chunking"]["context_lines"] = context_lines
    compatible["compatible_chunkers"] = ["line", "para"]
    assert validate_preset(compatible)["chunking"]["context_lines"] == context_lines

    para_only = deepcopy(compatible)
    para_only["name"] = "para-only"
    para_only["compatible_chunkers"] = ["para"]
    path = _write_preset(tmp_path, para_only, "para-only")
    stderr = io.StringIO()
    assert (
        main(["preset", "validate", str(path)], stdout=io.StringIO(), stderr=stderr)
        == 64
    )
    assert "chunking contains unknown fields" in stderr.getvalue()


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


def test_wire_fields_pass_through_and_render_once(tmp_path: Path) -> None:
    data = deepcopy(dict(resolve_preset("jgrep").data))
    data["name"] = "wire-fields"
    data["questions"]["matches_query"]["wire_hint"] = {"mode": "strict"}
    data["questions"]["matches_query"]["criteria"]["true"]["wire_note"] = "x"
    path = _write_preset(tmp_path, data, "wire-fields")
    seen = []
    stderr = io.StringIO()
    assert main(
        ["run", "--preset", str(path), "--query", "launch"],
        stdin=io.StringIO("launch\n"),
        stdout=io.StringIO(),
        stderr=stderr,
        judge_fn=lambda state, questions, model: (
            seen.append(questions)
            or JudgeResponse({"matches_query": NoulAnswer(0.9)})
        ),
        cache_store=CacheStore(tmp_path / "cache"),
    ) == 0
    assert seen[0]["matches_query"]["wire_hint"] == {"mode": "strict"}
    assert seen[0]["matches_query"]["criteria"]["true"]["wire_note"] == "x"
    assert stderr.getvalue().count("unknown_wire_field") == 2


def test_calibration_renders_preset_diagnostics_once(tmp_path: Path) -> None:
    data = deepcopy(dict(resolve_preset("jgrep").data))
    data["name"] = "calibration-warning"
    data["questions"]["matches_query"]["wire_hint"] = {"mode": "strict"}
    path = _write_preset(tmp_path, data, "calibration-warning")
    preset = load_preset(path)
    wire_state = {"focus": "focus", "context": {"query": "query"}}
    request = build_canonical_request(
        wire_state, preset.questions, model=preset.model
    )
    cache_dir = tmp_path / "cache"
    CacheStore(cache_dir).publish(
        wire_state,
        JudgeResponse({"matches_query": NoulAnswer(0.9)}),
        battery=preset.questions,
        preset=preset.name,
        preset_version=preset.version,
        configured_model=preset.model,
        transport_identity=request.transport_identity,
        state_ref="state#1",
    )
    stderr = io.StringIO()
    assert (
        main(
            [
                "calibrate",
                "--preset",
                str(path),
                "--cache-dir",
                str(cache_dir),
            ],
            stdout=io.StringIO(),
            stderr=stderr,
            judge_fn=lambda *_: JudgeResponse(
                {"matches_query": NoulAnswer(0.9)}
            ),
        )
        == 0
    )
    assert stderr.getvalue().count("unknown_wire_field") == 1


def test_cache_hash_envelope_rejects_unknown_transport_fields(tmp_path: Path) -> None:
    preset = resolve_preset("jgrep")
    wire_state = {"focus": "focus", "context": {"query": "query"}}
    request = build_canonical_request(
        wire_state, preset.questions, model=preset.model
    )
    store = CacheStore(tmp_path / "cache")
    entry = store.publish(
        wire_state,
        JudgeResponse({"matches_query": NoulAnswer(0.9)}),
        battery=preset.questions,
        preset=preset.name,
        preset_version=preset.version,
        configured_model=preset.model,
        transport_identity=request.transport_identity,
        state_ref="state#1",
    )
    payload = entry.to_dict()
    payload["transport_identity"]["unknown"] = "field"
    with pytest.raises(ValueError, match="invalid transport identity"):
        _parse_entry(payload, entry.cache_key, preset.questions)


def test_closed_blocks_reject_unknown_fields_and_invalid_coverage() -> None:
    for field in ("chunking", "thresholds", "output"):
        data = deepcopy(dict(resolve_preset("jgrep").data))
        if field == "chunking":
            data[field]["unknown"] = True
        elif field == "thresholds":
            data[field]["matches_query"]["unknown"] = True
        else:
            data[field]["unknown"] = True
        with pytest.raises(PresetValidationError):
            validate_preset(data)

    limits = deepcopy(dict(resolve_preset("jgrep").data))
    limits["chunking"]["limits"]["unknown"] = True
    with pytest.raises(PresetValidationError):
        validate_preset(limits)

    with pytest.raises(ValueError):
        compile_policy("any(matches_query.noul ~= 0.75)", resolve_preset("jgrep"))

    with pytest.raises(ValueError, match="unknown coverage reason"):
        CoverageRecord(
            "partial",
            {"discovered": 1, "judged": 0, "emitted": 0, "skipped": 1, "failed": 0},
            ("unknown",),
            RecordMeta("jgrep", "1", "typesafe-ai/jev", "para", "not_applicable"),
        )
    with pytest.raises(ValueError, match="five required fields"):
        CoverageRecord(
            "partial",
            {"unknown": 1},
            (),
            RecordMeta("jgrep", "1", "typesafe-ai/jev", "para", "not_applicable"),
        )


@pytest.mark.parametrize("level_count", [3, 4])
def test_v3_score_schema_accepts_wire_supported_level_counts(level_count: int) -> None:
    data = deepcopy(dict(resolve_preset("diff-risk-heat").data))
    data["schema"] = "jm.preset/v3"
    data["parameters"] = {"declared": []}
    data["questions"]["change_scope"]["criteria"] = deepcopy(
        data["questions"]["change_scope"]["criteria"][:level_count]
    )
    validate_preset(data)


@pytest.mark.parametrize("level_count", [2, 5])
def test_v3_score_schema_rejects_unsupported_level_counts(level_count: int) -> None:
    data = deepcopy(dict(resolve_preset("diff-risk-heat").data))
    data["schema"] = "jm.preset/v3"
    data["parameters"] = {"declared": []}
    criteria = deepcopy(data["questions"]["change_scope"]["criteria"])
    if level_count == 5:
        criteria.append(deepcopy(criteria[-1]))
    data["questions"]["change_scope"]["criteria"] = criteria[:level_count]
    with pytest.raises(PresetValidationError, match="three or four"):
        validate_preset(data)


@pytest.mark.parametrize(
    "criteria", [{"yes": {}}, {"true": {}, "false": {}, "maybe": {}}]
)
def test_noul_schema_rejects_exotic_polarities(criteria: dict[str, object]) -> None:
    data = deepcopy(dict(resolve_preset("jgrep").data))
    data["questions"]["matches_query"]["criteria"] = criteria
    with pytest.raises(PresetValidationError, match="true and false"):
        validate_preset(data)
