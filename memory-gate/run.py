#!/usr/bin/env python3
"""Offline/replayable memory-gate calibration runner.

The eval imports the production request builder/parser from ``harness/src``.
In a checkout (rather than an installed wheel), this bootstrap also adds the
sibling ``jm`` checkout to sys.path, matching harness/pyproject.toml's local
``jm`` editable dependency and its pausanias file dependency.  Candidate
regeneration is intentionally local-only; locomo regeneration is documented
below and must use pausanias' pinned runner/fetch, not a vendored dataset.

No command in this module performs a Jev request.  ``score`` consumes a
committed/cache response JSON file; a live adapter is supplied by the eventual
homelab invocation outside this offline contractor.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
for _path in (ROOT / "harness" / "src", ROOT / "jm", Path("/tmp/pausanias") / "src"):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import artifacts
import lock as lock_module
import metrics as metrics_module
import pipeline
import sweep as sweep_module

production_request = pipeline.build_request
parse_production_scores = pipeline.parse_scores
_production_adapter = pipeline.production_adapter
prepare_candidates = pipeline.prepare_candidates
select_blocks = pipeline.select_blocks
validate_scores = pipeline.validate_scores

BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260929
TAU_REFERENCE = 0.6
LOCOMO_URL = "https://raw.githubusercontent.com/snap-research/locomo/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json"
sha256_bytes = artifacts.sha256
canonical_json = artifacts.canonical_json


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# Consolidated helpers — import from authoritative modules, don't copy.
wilson = metrics_module.wilson
ratio = metrics_module.ratio

# Public runner API is the complete §5 implementation, not the historical stub.
metrics = metrics_module.metrics


def filter_locomo_category5(dataset: Sequence[Mapping[str, Any]], *, expected_conversations: int = 10, expected_questions: int = 446) -> list[dict[str, Any]]:
    """Keep category-5 questions and refuse partial locomo10 datasets."""
    filtered: list[dict[str, Any]] = []
    question_count = 0
    for conversation in dataset:
        questions = conversation.get("qa", conversation.get("questions", []))
        if not isinstance(questions, list):
            continue
        kept = [q for q in questions if str(q.get("category", q.get("cat", ""))) == "5"]
        if kept:
            copy = dict(conversation)
            copy["qa"] = kept
            filtered.append(copy)
            question_count += len(kept)
    if len(filtered) != expected_conversations or question_count != expected_questions:
        raise ValueError(f"locomo category-5 assertion failed: {len(filtered)} conversations / {question_count} questions")
    return filtered


def safety_cases(dataset: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Adapt the LOCOMO benchmark export after its rendering/ingestion stage.

    The pinned pausanias runner entry points are ``load_dataset`` for validation,
    ``_render_entries``/``render_session`` for markdown ingestion, and its
    ``search``/index path for retrieval.  This adapter intentionally accepts
    only the resulting per-question ``retrieved`` records; it never invents a
    second retrieval implementation.

    Retrieval enforcement (Blocker 2):
    - Every question MUST have a ``retrieved`` field (list).  A missing field
      means retrieval never ran and the dataset cannot be used for safety.
    - A question with an empty ``retrieved`` list is accepted only if it also
      carries ``retrieval_provenance`` proving the pinned pausanias pipeline
      produced a genuine zero-retrieval result.
    """
    filtered = filter_locomo_category5(dataset)
    cases: list[dict[str, Any]] = []
    for conversation_index, conversation in enumerate(filtered):
        for question_index, question in enumerate(conversation["qa"]):
            query = question.get("question", question.get("query"))
            if not isinstance(query, str):
                raise TypeError("LOCOMO benchmark export question lacks query string")
            # Retrieval enforcement: refuse questions where retrieval never ran
            if "retrieved" not in question and "candidates" not in question:
                raise ValueError(
                    f"LOCOMO question {conversation_index}-{question_index} has no "
                    f"'retrieved' field — retrieval never ran; dataset unusable for safety"
                )
            retrieved = question.get("retrieved", question.get("candidates"))
            if not isinstance(retrieved, list):
                raise TypeError("LOCOMO benchmark export question 'retrieved' must be a list")
            # Zero-candidate enforcement: distinguish "no candidates retrieved"
            # from "retrieval never ran" by requiring retrieval_provenance
            if len(retrieved) == 0:
                provenance = question.get("retrieval_provenance")
                if not isinstance(provenance, dict) or "pipeline" not in provenance:
                    raise ValueError(
                        f"LOCOMO question {conversation_index}-{question_index} has zero "
                        f"retrieval candidates but no retrieval_provenance — cannot "
                        f"distinguish 'no candidates retrieved' from 'retrieval never ran'"
                    )
            cases.append({"case_id": f"locomo-{conversation_index}-{question_index}",
                          "query": query, "retrieved": retrieved, "answerable": False,
                          "retrieval_provenance": question.get("retrieval_provenance")})
    if len(cases) != 446:
        raise ValueError(f"LOCOMO safety export has {len(cases)} questions, expected 446")
    return cases


def candidate_rows(cases: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        retrieved = case.get("retrieved", case.get("candidates", []))
        selected = prepare_candidates(case, retrieved)
        sources: dict[str, str] = {}
        for item in retrieved:
            if isinstance(item, Mapping) and isinstance(item.get("excerpt"), str) and item["excerpt"]:
                sources.setdefault(pipeline.content_hash(item["excerpt"]), item["excerpt"])
        for rank, item in enumerate(selected):
            source = sources.get(str(item["content_hash"]))
            if source is None or source[:pipeline.EXCERPT_CHARS] != item["excerpt"]:
                raise ValueError("selected candidate source changed during artifact generation")
            rows.append({"case_id": case["case_id"], "candidate_id": f"{case['case_id']}:{item['id']}", "query": case["query"], "path": item["path"], "heading": item["heading"], "rank": rank, "untruncated_excerpt_hash": item["content_hash"], "presented_excerpt": item["excerpt"], **provenance})
    return rows


def validate_artifact(run_dir: Path) -> None:
    artifacts.validate_run(run_dir)


def lock_witness(lock_path: Path, calibration_paths: Sequence[Path], tau: float) -> dict[str, Any]:
    paths = tuple(calibration_paths)
    run_dir = lock_path.parent
    stream_names = {path.name for path in paths}
    if {"candidates.jsonl", "scores.jsonl", "labels.jsonl"} <= stream_names:
        report = run_dir / "report.md"
        paths = (*paths, report) if report not in paths else paths
        validate_artifact(run_dir)
    return lock_module.write_lock(lock_path, paths, tau)


def verify_witness(lock: Path, witness: str, remote: str = "origin", *, safety_outputs: Sequence[Path] = (), repo: Path = ROOT) -> str:
    # Preserve the historical runner API while exposing lock.py predicates.
    try:
        return lock_module.verify_witness(lock, witness, remote=remote, safety_outputs=safety_outputs, repo=repo)
    except lock_module.WitnessError as exc:
        raise ValueError(str(exc)) from exc


def load_cases(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("cases", data.get("questions", []))
    if not isinstance(data, list):
        raise TypeError("case file must contain a list or cases field")
    return [dict(case) for case in data if isinstance(case, Mapping)]


def _resolve_scope_flags(case: Mapping[str, Any]) -> list[str]:
    """Derive pausanias --project / --all-projects flags from case scope.

    Case schema:
      scope.project  -> --project <name>
      scope.all_projects true (or absent scope / absent scope.project) -> --all-projects
    """
    scope = case.get("scope")
    if isinstance(scope, Mapping):
        project = scope.get("project")
        if isinstance(project, str) and project:
            return ["--project", project]
        if scope.get("all_projects"):
            return ["--all-projects"]
    # No scope block at all -> all-projects (backward-compatible default)
    return ["--all-projects"]


def _search_pausanias(case: Mapping[str, Any], *, executable: str, config: Path | None,
                      cwd: Path | None = None) -> list[dict[str, Any]]:
    """Run the pinned pausanias CLI; stdout is deliberately the only protocol."""
    command = [executable, "-m", "pausanias"]
    if config is not None:
        command += ["--config", str(config)]
    command += ["search", "--json"]
    command += ["--retrieval-mode", "fused"]
    command += _resolve_scope_flags(case)
    command += [str(case.get("query", ""))]
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    if isinstance(payload, dict):
        payload = payload.get("items", payload.get("results", payload.get("candidates", [])))
    if not isinstance(payload, list):
        raise TypeError("pausanias --json output must be a list or results object")
    return [dict(item) for item in payload if isinstance(item, Mapping)]


def _corpus_fingerprint(config_path: Path | None) -> str:
    """Deterministic corpus fingerprint: SHA-256 of the config file bytes.

    When no config is available, returns 'unavailable' rather than the
    previous 'unknown' — honesty over placeholders.
    """
    if config_path is not None and config_path.exists():
        return sha256_bytes(config_path.read_bytes())
    return "unavailable"


def _git_revision(repo: Path | None, label: str = "revision") -> str:
    """Best-effort git rev-parse HEAD for a checkout; 'unavailable' on failure."""
    if repo is None:
        return "unavailable"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo,
            capture_output=True, text=True, timeout=5, check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unavailable"


# Maximum fraction of zero-candidate cases before generation fails.
# DESIGN §4 spirit: "silent mass-emptiness must be impossible."
# If >20% of cases retrieve zero candidates, the run is refused.
EMPTY_RETRIEVAL_THRESHOLD = 0.20


class EmptyRetrievalError(RuntimeError):
    """Raised when too many cases retrieve zero candidates."""


def generate_candidates(case_path: Path, output: Path, *, pausanias_executable: str = sys.executable,
                         pausanias_config: Path | None = None, pausanias_cwd: Path | None = None,
                         provenance: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    cases = load_cases(case_path)
    base = dict(provenance or {})
    base.setdefault("case_set_fingerprint", sha256_bytes(case_path.read_bytes()))
    base.setdefault("corpus_fingerprint", _corpus_fingerprint(pausanias_config))
    pausanias_checkout = Path("/tmp/pausanias")
    base.setdefault("pausanias_revision", _git_revision(
        pausanias_checkout if pausanias_checkout.is_dir() else None))
    base.setdefault("retrieval_config", {
        "config": str(pausanias_config) if pausanias_config else None,
        "retrieval_mode": "fused",
        "command": "python -m pausanias --config <config> search --json --retrieval-mode fused",
    })
    base.setdefault("production_builder_hash", artifacts.authoritative_production_builder_hash())
    base.setdefault("configured_model_id", "unscored")
    base.setdefault("served_model_id", "unscored")
    base.setdefault("harness_revision", _git_revision(ROOT / "harness" if (ROOT / "harness").is_dir() else None))
    base.setdefault("jm_revision", _git_revision(ROOT / "jm" if (ROOT / "jm").is_dir() else None))
    enriched = []
    empty_count = 0
    for case in cases:
        current = dict(case)
        current.setdefault("case_id", current.get("id"))
        if not isinstance(current["case_id"], str):
            raise TypeError("each case requires id/case_id")
        retrieved = _search_pausanias(case, executable=pausanias_executable, config=pausanias_config, cwd=pausanias_cwd)
        current["retrieved"] = retrieved
        # Record per-case retrieval config in provenance
        scope_flags = _resolve_scope_flags(case)
        current["retrieval_config"] = {
            "retrieval_mode": "fused",
            "scope": scope_flags,
        }
        if not retrieved:
            empty_count += 1
        enriched.append(current)
    # Guard: refuse generation if too many cases are empty (DESIGN §4)
    if cases and empty_count / len(cases) > EMPTY_RETRIEVAL_THRESHOLD:
        raise EmptyRetrievalError(
            f"{empty_count}/{len(cases)} cases ({empty_count / len(cases):.0%}) "
            f"retrieved zero candidates — exceeds {EMPTY_RETRIEVAL_THRESHOLD:.0%} "
            f"threshold; refusing generation (scope or config likely misconfigured)"
        )
    rows = candidate_rows(enriched, base)
    for case_id in {row["case_id"] for row in rows}:
        group = [row for row in rows if row["case_id"] == case_id]
        digest = artifacts._canonical_hash(group)
        for row in group:
            row["canonical_request_hash"] = digest
    artifacts.write_candidates(output, rows)
    return rows


def _client_response(client: Any, request: Mapping[str, Any], *, model: str) -> Mapping[str, Any]:
    """Cross the one production-adapter seam and retain response metadata."""
    response = pipeline.evaluate_production(request, client, model=model)
    if not isinstance(response, Mapping):
        raise TypeError("production adapter response must be a mapping")
    return response


def _formed_state(request: Mapping[str, Any]) -> Any:
    from jm.client import State
    return State("harness", json.dumps(request["state"], ensure_ascii=False, sort_keys=True))


def _cache_key(request: Mapping[str, Any], model: str) -> str:
    """Use jm's canonical v3 envelope from the formed/projected State."""
    try:
        from jm.cache import build_cache_preimage, cache_key
        envelope = build_cache_preimage(
            state=_formed_state(request), questions=request["questions"], model=model,
        )
        return cache_key(envelope)
    except (ImportError, KeyError, TypeError, ValueError):
        return fingerprint({
            "cache_schema": "memory-gate-adapter/v1",
            "request_schema": 1,
            "protocol_version": "0.0.1",
            "model": model,
            "adapter_version": "memory-gate-production-adapter/v1",
            "request": request,
        })


def score_cases(cases: Sequence[Mapping[str, Any]], client: Any, *, model: str,
                cache_dir: Path | None = None, bypass_cache: bool = False) -> list[dict[str, Any]]:
    """Score the exact production batteries, with a small content-addressed replay cache."""
    rows: list[dict[str, Any]] = []
    provenance_defaults = {
        "case_set_fingerprint": "runtime", "corpus_fingerprint": "runtime",
        "pausanias_revision": "runtime", "retrieval_config": {"mode": "production"},
        "production_builder_hash": artifacts.authoritative_production_builder_hash(),
        "configured_model_id": None, "served_model_id": None,
        "harness_revision": "runtime", "jm_revision": "runtime",
    }
    for case in cases:
        candidates = prepare_candidates(case, case.get("retrieved", case.get("candidates", [])))
        request = pipeline.build_request(str(case["query"]), candidates)
        key = _cache_key(request, model)
        cache_file = cache_dir / f"{key.removeprefix('sha256:')}.json" if cache_dir else None
        error = None
        configured_model = model
        served_model = model
        try:
            if cache_file and cache_file.exists() and not bypass_cache:
                response = json.loads(cache_file.read_text())
            else:
                response = _client_response(client, request, model=model)
                if cache_file:
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    cache_file.write_text(json.dumps(response, sort_keys=True))
            configured_raw = response.get("configured_model")
            served_raw = response.get("served_model")
            if not configured_raw or not served_raw:
                raise ValueError("authoritative response omitted verified model identity")
            configured_model = str(configured_raw)
            served_model = str(served_raw)
            if configured_model != model:
                raise ValueError(f"configured model {configured_model!r} differs from requested {model!r}")
            if served_model in {"unknown", "unscored", "configured", "None"}:
                raise ValueError("authoritative response omitted served model identity")
            scores = pipeline.parse_scores(response, candidates)
        except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:  # request errors invalidate the run
            scores, error = {}, f"{type(exc).__name__}: {exc}"
        if not candidates:
            rows.append({"case_id": case["case_id"], "candidates": []})
        for rank, candidate in enumerate(candidates):
            cid = str(candidate["id"])
            row = {"case_id": case["case_id"], "candidate_id": f"{case['case_id']}:{cid}",
                   "query": case["query"], "path": candidate["path"], "heading": candidate["heading"],
                   "rank": rank, "score": scores.get(cid), "request_error": error,
                   "coverage": error is None and cid in scores,
                   "untruncated_excerpt_hash": candidate["content_hash"], "presented_excerpt": candidate["excerpt"]}
            row.update({**provenance_defaults, "configured_model_id": configured_model,
                        "served_model_id": served_model})
            namespaced = [
                {**candidate, "id": f"{case['case_id']}:{candidate['id']}"}
                for candidate in candidates
            ]
            row["canonical_request_hash"] = fingerprint(
                pipeline.build_request(str(case["query"]), namespaced)
            )
            rows.append(row)
    return rows


def run_repeatability(cases: Sequence[Mapping[str, Any]], client: Any, *, model: str,
                      cache_dir: Path, tau: float) -> dict[str, Any]:
    strata = {"abstain": [], "verbatim": [], "paraphrase": []}
    for case in cases:
        category = str(case.get("category", ""))
        if category in strata:
            strata[category].append(case)
    if any(len(values) < 4 for values in strata.values()):
        raise ValueError("repeatability requires four cases in each category stratum")
    subset = [case for values in strata.values() for case in values[:4]]
    replicates = []
    for index in range(6):
        rows = score_cases(subset, client, model=model, cache_dir=cache_dir, bypass_cache=index > 0)
        identities = {(row.get("configured_model_id"), row.get("served_model_id")) for row in rows}
        if len(identities) != 1 or any(c is None or s is None for c, s in identities):
            raise ValueError("repeatability refuses missing or mixed configured/served model identities")
        configured_model, served_model = next(iter(identities))
        replicates.append({"replicate": index, "configured_model_id": configured_model,
                           "served_model_id": served_model, "scores": rows,
                           "cache_bypassed": index > 0})
    if len(replicates) != 6:
        raise ValueError("repeatability requires exactly six replicates")
    expected_ids = {str(row["candidate_id"]) for row in replicates[0]["scores"]}
    for replicate in replicates:
        rows = replicate.get("scores", [])
        ids = {str(row["candidate_id"]) for row in rows}
        if len(rows) != len(expected_ids) or ids != expected_ids or any(
            row.get("score") is None or row.get("request_error") for row in rows
        ):
            raise ValueError("repeatability refuses missing or errored scores")
    result = repeatability(replicates, tau)
    result["strata"] = {name: [case.get("case_id", case.get("id")) for case in values[:4]]
                         for name, values in strata.items()}
    return result


def repeatability(replicates: Sequence[Mapping[str, Any]], tau: float) -> dict[str, Any]:
    if not replicates:
        raise ValueError("repeatability requires replicate 0")
    identities = set()
    for replicate in replicates:
        rows = replicate.get("scores", replicate.get("rows", []))
        row_identities = {(row.get("configured_model_id"), row.get("served_model_id")) for row in rows}
        if len(row_identities) != 1 or any(c is None or s is None for c, s in row_identities):
            raise ValueError("repeatability refuses missing or mixed configured/served model identities")
        identity = next(iter(row_identities))
        declared = (replicate.get("configured_model_id"), replicate.get("served_model_id"))
        if declared != identity:
            raise ValueError("repeatability refuses mixed configured/served model identities")
        identities.add(identity)
    if len(identities) != 1:
        raise ValueError("repeatability refuses mixed configured/served model identities")
    grouped: dict[str, list[float]] = {}
    for replicate in replicates:
        for row in replicate.get("scores", replicate.get("rows", [])):
            if row.get("score") is not None:
                grouped.setdefault(str(row["candidate_id"]), []).append(float(row["score"]))
    details = {}
    crossing = 0
    for cid, values in grouped.items():
        spread = max(values) - min(values)
        details[cid] = {"stddev": statistics.pstdev(values), "worst_spread": spread,
                         "crosses_tau": min(values) <= tau < max(values)}
        crossing += details[cid]["crosses_tau"]
    return {"replicates": len(replicates), "tau": tau, "model_identity": list(next(iter(identities))),
            "per_candidate": details, "worst_spread": max((x["worst_spread"] for x in details.values()), default=0),
            "fraction_crossing_tau": ratio(crossing, len(details))}


def freeze_safety(path: Path, *, witness_commit: str, tau: float, false_injections: int,
                  total: int, repeatability_data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if total < 446:
        raise ValueError("authoritative safety freeze requires at least 446 cases")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen safety artifact: {path}")
    upper = wilson(false_injections, total)[1] if total else None
    artifact = {"authoritative": True, "witness_commit": witness_commit, "tau": tau,
                "false_injections": false_injections, "total": total,
                "point_estimate": ratio(false_injections, total), "wilson_95": wilson(false_injections, total),
                "accepts": total > 0 and false_injections / total <= .02 and upper <= .04,
                "repeatability": repeatability_data}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x") as stream:
            json.dump(artifact, stream, indent=2, sort_keys=True); stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return artifact


def safety_result(scores: Sequence[Mapping[str, Any]], tau: float, *, witness_commit: str,
                  output: Path, repeatability_data: Mapping[str, Any] | None = None,
                  expected_case_ids: set[str] | None = None,
                  scores_output: Path | None = None) -> dict[str, Any]:
    """Freeze category-5 case-level false injection after witness verification."""
    observed_case_ids: set[str] = set()
    by_case: dict[str, list[Mapping[str, Any]]] = {
        str(case_id): [] for case_id in (expected_case_ids or set())
    }
    for row in scores:
        observed_case_ids.add(str(row["case_id"]))
        case_id = str(row["case_id"])
        by_case.setdefault(case_id, [])
        # A retrieval miss is represented by a case marker with no candidates;
        # it remains a denominator observation but has no rows to validate.
        if row.get("candidates") == []:
            continue
        by_case[case_id].append(row)
    if expected_case_ids is not None and observed_case_ids != set(expected_case_ids):
        missing = sorted(set(expected_case_ids) - observed_case_ids)
        extra = sorted(observed_case_ids - set(expected_case_ids))
        raise ValueError(f"safety scores have incomplete case IDs (missing={missing}, extra={extra})")
    false_injections = 0
    for rows in by_case.values():
        if not rows:
            continue
        candidates = [
            {"id": row["candidate_id"], "path": row["path"],
             "heading": row["heading"], "excerpt": row["presented_excerpt"]}
            for row in rows
        ]
        scores_by_id = {row["candidate_id"]: row.get("score") for row in rows}
        if any(row.get("coverage") is not True for row in rows):
            raise ValueError("safety scores have incomplete production coverage")
        if select_blocks(candidates, scores_by_id, tau):
            false_injections += 1
    total = len(expected_case_ids) if expected_case_ids is not None else len(by_case)
    if scores_output is None:
        return freeze_safety(output, witness_commit=witness_commit, tau=tau,
                             false_injections=false_injections, total=total,
                             repeatability_data=repeatability_data)
    if output.exists() or scores_output.exists():
        raise FileExistsError("refusing to overwrite frozen safety outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    scores_output.parent.mkdir(parents=True, exist_ok=True)
    # Stage both artifacts in a temporary directory; publish via a single
    # directory rename so a failure after partial publication cannot strand
    # one artifact without the other (fix 4: failure-atomic publish).
    staging = output.parent / f".safety-staging-{os.getpid()}"
    staging.mkdir(exist_ok=False)
    staged_safety = staging / output.name
    staged_scores = staging / scores_output.name
    try:
        artifact = freeze_safety(staged_safety, witness_commit=witness_commit, tau=tau,
                                 false_injections=false_injections, total=total,
                                 repeatability_data=repeatability_data)
        staged_scores.write_text("".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in scores))
        # Move both out of staging into their final locations.  If the second
        # rename fails, remove the first so no partial outputs remain.
        os.replace(staged_scores, scores_output)
        try:
            os.replace(staged_safety, output)
        except BaseException:
            scores_output.unlink(missing_ok=True)
            raise
        return artifact
    finally:
        # Clean up any staging remnants.
        staged_safety.unlink(missing_ok=True)
        staged_scores.unlink(missing_ok=True)
        try:
            staging.rmdir()
        except OSError:
            pass


def posthoc_curve(run_dir: Path, curve: Sequence[Mapping[str, Any]]) -> None:
    safety = run_dir / "safety.json"
    scores = run_dir / "safety-scores.jsonl"
    if not safety.exists() or not scores.exists():
        raise ValueError("posthoc safety curve requires frozen safety artifacts")
    payload = json.loads(safety.read_text())
    _validate_frozen_safety_schema(payload, run_dir)
    write_posthoc_curve(run_dir / "posthoc-safety-curve.json", curve)


def _validate_frozen_safety_schema(payload: Mapping[str, Any], run_dir: Path) -> None:
    """Complete frozen-safety schema validation (fix 5).

    Validates types for point_estimate/counts, cross-checks counts vs scores,
    recomputes Wilson bounds and acceptance, checks repeatability block shape,
    witness hash format, and tau == LOCK tau.
    """
    required = {"authoritative", "witness_commit", "tau", "false_injections", "total",
                "point_estimate", "wilson_95", "accepts", "repeatability"}
    if set(payload) != required or payload.get("authoritative") is not True:
        raise ValueError("posthoc safety curve requires validated frozen safety.json")
    # Witness hash format
    if not isinstance(payload["witness_commit"], str) or not payload["witness_commit"]:
        raise ValueError("posthoc safety curve requires validated frozen safety.json")
    if len(payload["witness_commit"]) < 7:
        raise ValueError("posthoc safety curve has invalid witness hash format")
    # tau
    if (isinstance(payload["tau"], bool) or not isinstance(payload["tau"], (int, float))
            or not math.isfinite(payload["tau"])):
        raise ValueError("posthoc safety curve has invalid tau")
    lock_path = run_dir / "LOCK.json"
    if lock_path.exists() and payload["tau"] != json.loads(lock_path.read_text()).get("tau"):
        raise ValueError("posthoc safety curve tau differs from LOCK.json")
    # Counts — types and cross-check
    if (isinstance(payload["total"], bool) or not isinstance(payload["total"], int)
            or payload["total"] < 446 or isinstance(payload["false_injections"], bool)
            or not isinstance(payload["false_injections"], int)):
        raise ValueError("posthoc safety curve has invalid totals")
    if payload["false_injections"] < 0 or payload["false_injections"] > payload["total"]:
        raise ValueError("posthoc safety curve has invalid false_injections vs total")
    # point_estimate type and cross-check
    pe = payload["point_estimate"]
    if isinstance(pe, bool) or not isinstance(pe, (int, float)) or not math.isfinite(pe):
        raise ValueError("posthoc safety curve has invalid point_estimate type")
    expected_pe = payload["false_injections"] / payload["total"]
    if abs(pe - expected_pe) > 1e-12:
        raise ValueError("posthoc safety curve point_estimate inconsistent with counts")
    # Wilson bounds — recompute and compare
    if not isinstance(payload["wilson_95"], list) or len(payload["wilson_95"]) != 2:
        raise ValueError("posthoc safety curve has invalid Wilson fields")
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool)
               and math.isfinite(value) for value in payload["wilson_95"]):
        raise ValueError("posthoc safety curve has invalid Wilson fields")
    expected_wilson = wilson(payload["false_injections"], payload["total"])
    if expected_wilson is not None and (
            abs(payload["wilson_95"][0] - expected_wilson[0]) > 1e-10
            or abs(payload["wilson_95"][1] - expected_wilson[1]) > 1e-10):
        raise ValueError("posthoc safety curve Wilson bounds inconsistent with counts")
    # Acceptance — recompute and compare
    if not isinstance(payload["accepts"], bool):
        raise TypeError("posthoc safety curve has invalid acceptance")
    expected_accepts = (payload["total"] > 0
                        and payload["false_injections"] / payload["total"] <= .02
                        and payload["wilson_95"][1] <= .04)
    if payload["accepts"] != expected_accepts:
        raise ValueError("posthoc safety curve acceptance inconsistent with counts/Wilson")
    # Repeatability block shape
    rep = payload.get("repeatability")
    if rep is not None:
        if not isinstance(rep, Mapping):
            raise ValueError("posthoc safety curve has invalid repeatability block shape")
        for required_key in ("replicates", "tau", "model_identity", "per_candidate",
                             "worst_spread", "fraction_crossing_tau"):
            if required_key not in rep:
                raise ValueError(f"posthoc safety curve repeatability missing {required_key}")


def write_posthoc_curve(path: Path, curve: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    path.write_text(json.dumps({"authoritative": False, "curve": list(curve)}, indent=2, sort_keys=True) + "\n")


def _repo_root(run_dir: Path) -> Path:
    """Compute repo root via git rev-parse --show-toplevel (Blocker 3).

    Falls back to walking parents for .git if git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=run_dir, capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
    except OSError:
        pass
    # Fallback: walk parents looking for .git
    for parent in (run_dir, *run_dir.parents):
        if (parent / ".git").exists():
            return parent
    return run_dir.parent


def _format_report(result: dict[str, Any]) -> str:
    """Format sweep results into the authoritative report.md content."""
    lines = [
        "RESULTS — memory-gate metrics", "",
        "Thresholds: " + ", ".join(f"{x:.6g}" for x in result["thresholds"]), "",
        "tau | any-injection | recall | precision | exact-packet | forbidden | retention | ROC-AUC | PR-AUC | Brier",
        "--- | --- | --- | --- | --- | --- | --- | --- | --- | ---",
    ]
    for row in result["table"]:
        def fmt(value):
            return "null" if value is None else f"{value:.6g}"
        lines.append(" | ".join([
            fmt(row[k]) for k in (
                "tau", "any_injection_rate", "packet_recall", "packet_precision",
                "exact_packet_rate", "forbidden_injection_rate", "retention",
                "roc_auc", "pr_auc", "brier",
            )
        ]))
    lines += ["", "Per-stratum any-injection (at each tau):"]
    for row in result["table"]:
        strata = ", ".join(
            f"{name}={value['cases']}/{value['total']} ({value['rate']!r})"
            for name, value in row["any_injection_rate_by_stratum"].items()
        )
        lines.append(f"tau={row['tau']:.6g}: {strata}")
    lines += ["", "Bootstrap 95% CIs (10,000 resamples; seed 20260929; percentile):"]
    for row in result["table"]:
        entries = []
        for name in (
            "any_injection_rate", "packet_recall", "packet_precision",
            "exact_packet_rate", "forbidden_injection_rate",
        ):
            bootstrap = row["bootstrap"][name]
            ci = bootstrap["ci"]
            ci_text = (
                "null" if ci is None
                else "[" + ", ".join(f"{value:.6g}" for value in ci) + "]"
            )
            entries.append(f"{name}={ci_text}; null-replicates={bootstrap['null_replicates']}")
        lines.append(f"tau={row['tau']:.6g}: " + " | ".join(entries))
    lines += ["", "Pareto frontier (abstain any-injection, packet recall):"]
    lines += [
        f"tau={r['tau']:.6g} abstain="
        f"{r['any_injection_rate_by_stratum'].get('abstain', {}).get('rate')!r} "
        f"recall={r['packet_recall']!r}"
        for r in result["pareto"]
    ]
    lines += ["", "Selection: " + json.dumps(result["selection"], sort_keys=True)]
    return "\n".join(lines) + "\n"


def _validate_labels_filled(run_dir: Path) -> None:
    """Refuse to score if labels are absent or still template-empty.

    Policy decision (Blocker 1, DESIGN §4): calibration scoring requires
    human-filled labels.  A label file with any null labels is considered
    template-empty and rejected.
    """
    labels_path = run_dir / "labels.jsonl"
    if not labels_path.exists():
        raise ValueError(
            "calibration scoring requires labels.jsonl — "
            "run label-template and fill labels first"
        )
    labels = read_jsonl(labels_path)
    if not labels:
        raise ValueError("labels.jsonl is empty")
    for row in labels:
        if row.get("label") is None:
            raise ValueError(
                "labels.jsonl contains unfilled template labels (null) — "
                "fill all labels before scoring"
            )
        if row.get("label") not in {"positive", "negative", "ambiguous"}:
            raise ValueError(
                f"labels.jsonl contains invalid label: {row.get('label')!r}"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    # -- candidates --
    p_candidates = sub.add_parser("candidates")
    p_candidates.add_argument("--cases", required=True, type=Path)
    p_candidates.add_argument("--output", required=True, type=Path)
    p_candidates.add_argument("--lane", default="calibration")
    p_candidates.add_argument("--pausanias-python", default=sys.executable)
    p_candidates.add_argument("--config", type=Path, default=None)

    # -- label-template --
    p_label = sub.add_parser("label-template")
    p_label.add_argument("--candidates", type=Path, default=Path("memory-gate/runs/fixture-dev"), dest="candidates_dir")

    # -- score --
    p_score = sub.add_parser("score")
    p_score.add_argument("--lane", default="calibration")
    p_score.add_argument("--run", required=True, type=Path, dest="run_dir")
    p_score.add_argument("--cases", type=Path, default=None)
    p_score.add_argument("--responses", type=Path, default=None)
    p_score.add_argument("--model", default="typesafe-ai/jev")
    p_score.add_argument("--witness", default=None)
    p_score.add_argument("--remote", default="origin")

    # -- report --
    p_report = sub.add_parser("report")
    p_report.add_argument("--run", required=True, type=Path, dest="run_dir")

    # -- lock --
    p_lock = sub.add_parser("lock")
    p_lock.add_argument("--run", required=True, type=Path, dest="run_dir")
    p_lock.add_argument("--tau", type=float, default=TAU_REFERENCE)

    # -- posthoc-safety-curve --
    p_posthoc = sub.add_parser("posthoc-safety-curve")
    p_posthoc.add_argument("--run", required=True, type=Path, dest="run_dir")

    parser.epilog = "Safety verification fetches the configured remote with pruning before accepting a witness."

    args = parser.parse_args(argv)

    # ---- label-template ----
    if args.command == "label-template":
        for row in read_jsonl(args.candidates_dir / "candidates.jsonl"):
            output = dict(row)
            output["label"] = None
            print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0

    # ---- lock ----
    if args.command == "lock":
        p = args.run_dir
        lock_witness(
            p / "LOCK.json",
            [p / "candidates.jsonl", p / "scores.jsonl", p / "labels.jsonl", p / "report.md"],
            args.tau,
        )
        return 0

    # ---- score ----
    if args.command == "score":
        run_dir = args.run_dir
        lane = args.lane

        if lane == "safety":
            if args.witness is None:
                raise SystemExit("safety scoring requires --witness <commit>")
            repo = _repo_root(run_dir)
            if args.cases is None:
                # Legacy stub path: verify witness only
                verify_witness(
                    run_dir / "LOCK.json", args.witness,
                    remote=args.remote,
                    safety_outputs=[run_dir / "safety.json", run_dir / "safety-scores.jsonl"],
                    repo=repo,
                )
                raise SystemExit("safety scoring without --cases requires a live adapter")

            witness = verify_witness(
                run_dir / "LOCK.json", args.witness,
                remote=args.remote,
                safety_outputs=[run_dir / "safety.json", run_dir / "safety-scores.jsonl"],
                repo=repo,
            )
            if args.responses is None:
                raise SystemExit("offline score requires --responses JSON")
            response_data = json.loads(args.responses.read_text())
            responses = iter(response_data if isinstance(response_data, list) else [response_data])

            def file_client_safety(_request):
                try:
                    return next(responses)
                except StopIteration as exc:
                    raise RuntimeError("response fixture exhausted") from exc

            cases = safety_cases(load_cases(args.cases))
            scores = score_cases(
                cases, file_client_safety, model=args.model,
                cache_dir=run_dir / "cache", bypass_cache=True,
            )
            expected_case_ids = {str(case["case_id"]) for case in cases}
            safety_result(
                scores,
                float(json.loads((run_dir / "LOCK.json").read_text())["tau"]),
                witness_commit=witness,
                output=run_dir / "safety.json",
                scores_output=run_dir / "safety-scores.jsonl",
                expected_case_ids=expected_case_ids,
            )
            return 0

        # -- calibration lane --
        if lane == "calibration":
            if args.responses is None:
                raise SystemExit("offline score requires --responses JSON")

            # Blocker 1: CONSUME existing candidates, never regenerate
            candidates_path = run_dir / "candidates.jsonl"
            if not candidates_path.exists():
                raise SystemExit(
                    "calibration scoring requires existing candidates.jsonl — "
                    "run the candidates command first"
                )

            # Blocker 1: refuse to score if labels absent or template-empty
            _validate_labels_filled(run_dir)

            # Read existing candidates
            existing_candidates = read_jsonl(candidates_path)

            # Group candidates by case_id, preserving order
            cases_by_id: dict[str, list[dict[str, Any]]] = {}
            for row in existing_candidates:
                cases_by_id.setdefault(row["case_id"], []).append(row)

            response_data = json.loads(args.responses.read_text())
            responses = iter(response_data if isinstance(response_data, list) else [response_data])

            def file_client_cal(_request):
                try:
                    return next(responses)
                except StopIteration as exc:
                    raise RuntimeError("response fixture exhausted") from exc

            score_rows: list[dict[str, Any]] = []
            for cand_group in cases_by_id.values():
                # Build request from existing candidates (immutable stream)
                query = cand_group[0]["query"]
                request_candidates = [
                    {"id": row["candidate_id"], "path": row["path"],
                     "heading": row["heading"], "excerpt": row["presented_excerpt"]}
                    for row in cand_group
                ]
                request = pipeline.build_request(query, request_candidates)
                error = None
                configured_model = args.model
                served_model = args.model
                try:
                    response = file_client_cal(request)
                    configured_raw = response.get("configured_model")
                    served_raw = response.get("served_model")
                    if not configured_raw or not served_raw:
                        raise ValueError("response omitted verified model identity")
                    configured_model = str(configured_raw)
                    served_model = str(served_raw)
                    if configured_model != args.model:
                        raise ValueError(
                            f"configured model {configured_model!r} differs "
                            f"from requested {args.model!r}"
                        )
                    scores_map = pipeline.parse_scores(response, request_candidates)
                except (KeyError, OSError, TypeError, ValueError, RuntimeError) as exc:
                    scores_map, error = {}, f"{type(exc).__name__}: {exc}"

                for row in cand_group:
                    cid = row["candidate_id"]
                    score_row = dict(row)  # Copy all provenance from candidate
                    score_row["score"] = scores_map.get(cid)
                    score_row["request_error"] = error
                    score_row["coverage"] = error is None and cid in scores_map
                    # Blocker 1: carry verified model identities from response,
                    # not from the (possibly stale) candidate row
                    score_row["configured_model_id"] = configured_model
                    score_row["served_model_id"] = served_model
                    score_rows.append(score_row)

            # Write only scores.jsonl — do NOT overwrite labels.jsonl (Blocker 1)
            (run_dir / "scores.jsonl").write_text(
                "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in score_rows)
            )
            return 0

        # Generic offline scoring with --cases
        if args.cases is not None:
            if args.responses is None:
                raise SystemExit("offline score requires --responses JSON")
            response_data = json.loads(args.responses.read_text())
            responses = iter(response_data if isinstance(response_data, list) else [response_data])

            def file_client_gen(_request):
                try:
                    return next(responses)
                except StopIteration as exc:
                    raise RuntimeError("response fixture exhausted") from exc

            cases = load_cases(args.cases)
            scores = score_cases(
                cases, file_client_gen, model=args.model,
                cache_dir=run_dir / "cache",
            )
            provenance = {
                "case_set_fingerprint": "runtime",
                "corpus_fingerprint": "runtime",
                "pausanias_revision": "runtime",
                "retrieval_config": {"mode": "production"},
                "production_builder_hash": artifacts.authoritative_production_builder_hash(),
                "configured_model_id": args.model,
                "served_model_id": args.model,
                "harness_revision": "runtime",
                "jm_revision": "runtime",
            }
            candidates = candidate_rows(cases, provenance)
            artifacts.write_artifact(
                run_dir, candidates, scores,
                [{**row, "label": "negative"} for row in candidates], "",
            )
            return 0

        raise SystemExit("score requires --lane calibration/safety or --cases")

    # ---- report ----
    if args.command == "report":
        run_dir = args.run_dir
        # The three JSONL streams must exist; report.md is the output of this
        # command so it may not exist yet.  Full validate_run (including
        # report.md) is deferred to the lock command.
        for required_stream in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"):
            if not (run_dir / required_stream).exists():
                raise artifacts.ArtifactValidationError(
                    f"missing {required_stream} — run candidates, label, and score first"
                )
        cases = metrics_module.load_run(run_dir, require_report=False)
        result = sweep_module.sweep(cases)
        text = _format_report(result)

        # Blocker 4: report.md is authoritative; results.txt is a byproduct
        (run_dir / "report.md").write_text(text)
        (run_dir / "results.txt").write_text(text)

        # Now that report.md exists, full validation should pass
        validate_artifact(run_dir)

        print(text, end="")
        return 0

    # ---- posthoc-safety-curve ----
    if args.command == "posthoc-safety-curve":
        run_dir = args.run_dir
        lock_path = run_dir / "LOCK.json"
        if not lock_path.exists():
            raise SystemExit("posthoc safety curve requires LOCK.json")
        lock_data = json.loads(lock_path.read_text())
        score_path = run_dir / "safety-scores.jsonl"
        if not score_path.exists():
            raise SystemExit("posthoc safety curve requires frozen safety-scores.jsonl")
        scores = read_jsonl(score_path)
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in scores:
            if row.get("candidates") == [] or "score" not in row:
                by_case.setdefault(str(row["case_id"]), [])
                continue
            by_case.setdefault(str(row["case_id"]), []).append(row)
        boundaries = sorted({
            float(row["score"]) for row in scores
            if "score" in row and row.get("candidates") != []
        })
        curve = []
        for tau in boundaries + [float(lock_data["tau"])]:
            injected = 0
            for rows in by_case.values():
                if not rows:
                    continue
                candidates = [
                    {"id": r["candidate_id"], "path": r["path"],
                     "heading": r["heading"], "excerpt": r["presented_excerpt"]}
                    for r in rows
                ]
                if select_blocks(
                    candidates,
                    {r["candidate_id"]: r["score"] for r in rows},
                    tau,
                ):
                    injected += 1
            curve.append({
                "tau": tau, "false_injections": injected,
                "total": len(by_case), "rate": ratio(injected, len(by_case)),
            })
        posthoc_curve(run_dir, curve)
        return 0

    # ---- candidates ----
    if args.command == "candidates":
        lane = args.lane
        case_path = args.cases
        output = args.output
        if lane == "safety":
            dataset = load_cases(case_path)
            rows = candidate_rows(safety_cases(dataset), {
                "case_set_fingerprint": sha256_bytes(case_path.read_bytes()),
                "corpus_fingerprint": "locomo-pinned",
                "pausanias_revision": "pinned",
                "retrieval_config": {
                    "benchmark": "eval.benchmarks.locomo.run",
                    "mode": "production",
                },
                "production_builder_hash": artifacts.authoritative_production_builder_hash(),
                "configured_model_id": "unscored",
                "served_model_id": "unscored",
                "harness_revision": "pinned",
                "jm_revision": "pinned",
            })
            by_case: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                by_case.setdefault(row["case_id"], []).append(row)
            for group in by_case.values():
                digest = artifacts._canonical_hash(group)
                for row in group:
                    row["canonical_request_hash"] = digest
            artifacts.write_candidates(output, rows)
            return 0
        generate_candidates(
            case_path, output,
            pausanias_executable=args.pausanias_python,
            pausanias_config=args.config,
        )
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
