"""Immutable memory-gate artifact schema and all-or-nothing validation."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pipeline

SCHEMA_VERSION = 1
PROVENANCE_FIELDS = (
    "case_set_fingerprint", "corpus_fingerprint", "pausanias_revision",
    "retrieval_config", "rank", "untruncated_excerpt_hash",
    "presented_excerpt", "canonical_request_hash", "production_builder_hash",
    "configured_model_id", "served_model_id", "harness_revision", "jm_revision",
)
CANDIDATE_FIELDS = ("case_id", "candidate_id", "query", "path", "heading", "rank", *PROVENANCE_FIELDS)
SCORE_FIELDS = ("case_id", "candidate_id", "score", "request_error", "coverage", *PROVENANCE_FIELDS)
LABEL_FIELDS = (*CANDIDATE_FIELDS, "label")
LABELS = {"positive", "negative", "ambiguous"}
# Phase C frozen safety records add this field after witness verification.
WITNESS_FIELD = "witness_commit"
FROZEN_SAFETY_FIELDS = (*SCORE_FIELDS, WITNESS_FIELD)

class ArtifactValidationError(ValueError):
    """The complete run is unusable for gating; no partial credit is allowed."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(f"invalid {path.name}: {exc}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ArtifactValidationError(f"{path.name} contains a non-object record")
    return rows


def write_artifact(run_dir: Path, candidates: Iterable[Mapping[str, Any]], scores: Iterable[Mapping[str, Any]], labels: Iterable[Mapping[str, Any]], report: str = "") -> None:
    """Write all three JSONL streams using the same field definition as validation."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = {"candidates.jsonl": list(candidates), "scores.jsonl": list(scores), "labels.jsonl": list(labels)}
    for name, data in rows.items():
        (run_dir / name).write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in data))
    (run_dir / "report.md").write_text(report)


def _required(record: Mapping[str, Any], fields: Sequence[str], kind: str) -> None:
    missing = set(fields) - record.keys()
    if missing:
        raise ArtifactValidationError(f"{kind} {record.get('candidate_id', '?')}: missing {sorted(missing)}")


def _provenance(record: Mapping[str, Any], kind: str) -> None:
    _required(record, PROVENANCE_FIELDS, kind)
    if not isinstance(record["presented_excerpt"], str) or not record["presented_excerpt"]:
        raise ArtifactValidationError(f"{kind}: presented_excerpt must be a non-empty string")
    if not isinstance(record["heading"], list) or not all(isinstance(x, str) for x in record["heading"]):
        raise ArtifactValidationError(f"{kind}: heading must be a string list")
    if isinstance(record["rank"], bool) or not isinstance(record["rank"], int) or record["rank"] < 0:
        raise ArtifactValidationError(f"{kind}: invalid rank")
    for field in ("case_id", "candidate_id", "query", "path", "untruncated_excerpt_hash", "canonical_request_hash", "production_builder_hash", "configured_model_id", "served_model_id", "pausanias_revision", "harness_revision", "jm_revision"):
        if not isinstance(record[field], str) or not record[field]:
            raise ArtifactValidationError(f"{kind}: invalid {field}")
    if not isinstance(record["retrieval_config"], (dict, list, str, int, float, bool)):
        raise ArtifactValidationError(f"{kind}: invalid retrieval_config")


def _validate_rows(rows: list[dict[str, Any]], fields: Sequence[str], kind: str) -> None:
    for row in rows:
        _required(row, fields, kind)
        _provenance(row, kind)
        if kind == "candidate" and not isinstance(row["query"], str):
            raise ArtifactValidationError("candidate: query must be a string")
        if kind == "score":
            score = row["score"]
            if score is None or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ArtifactValidationError("invalid score")
            if not isinstance(row["coverage"], bool) or not row["coverage"]:
                raise ArtifactValidationError("coverage is false or malformed")
            if row["request_error"] not in (None, "", False):
                raise ArtifactValidationError("request error present")
        if kind == "label" and row["label"] not in LABELS:
            raise ArtifactValidationError("invalid label")


def _canonical_hash(candidates: list[dict[str, Any]]) -> str:
    # The request builder is the phase-A authority.  Candidate IDs are removed
    # from the hash input only by the builder itself, never by this validator.
    request = pipeline.build_request(candidates[0]["query"], [
        {"id": r["candidate_id"], "path": r["path"], "heading": r["heading"], "excerpt": r["presented_excerpt"]}
        for r in candidates
    ])
    return sha256(canonical_json(request))


def validate_run(run_dir: Path) -> None:
    required = {"candidates.jsonl", "scores.jsonl", "labels.jsonl", "report.md"}
    if not run_dir.is_dir() or not required <= {p.name for p in run_dir.iterdir()}:
        raise ArtifactValidationError(f"missing artifact files: {sorted(required - {p.name for p in run_dir.iterdir()}) if run_dir.exists() else sorted(required)}")
    candidates, scores, labels = (_read(run_dir / name) for name in ("candidates.jsonl", "scores.jsonl", "labels.jsonl"))
    if not candidates:
        raise ArtifactValidationError("empty candidates artifact")
    _validate_rows(candidates, CANDIDATE_FIELDS, "candidate")
    _validate_rows(scores, SCORE_FIELDS, "score")
    _validate_rows(labels, LABEL_FIELDS, "label")
    for name, rows in (("candidate", candidates), ("score", scores), ("label", labels)):
        ids = [r["candidate_id"] for r in rows]
        dup = [k for k, v in Counter(ids).items() if v > 1]
        if dup:
            raise ArtifactValidationError(f"duplicate {name} records: {dup}")
    candidate_ids = {r["candidate_id"] for r in candidates}
    for name, rows in (("score", scores), ("label", labels)):
        ids = {r["candidate_id"] for r in rows}
        if ids != candidate_ids:
            raise ArtifactValidationError(f"{name} records missing or extra candidates")
    invariant_fields = ("case_set_fingerprint", "corpus_fingerprint", "pausanias_revision", "retrieval_config", "production_builder_hash", "configured_model_id", "served_model_id", "harness_revision", "jm_revision")
    for field in invariant_fields:
        values = {json.dumps(r[field], sort_keys=True) for r in candidates}
        if len(values) != 1:
            raise ArtifactValidationError(f"mixed provenance/model identity in {field}")
    by_id = {r["candidate_id"]: r for r in candidates}
    for row in scores + labels:
        candidate = by_id[row["candidate_id"]]
        for field in PROVENANCE_FIELDS:
            if row[field] != candidate[field]:
                raise ArtifactValidationError(f"provenance drift in {field}")
    # Hash is over the full source when supplied; this makes truncation errors
    # observable while retaining the DESIGN's compact committed representation.
    for row in candidates:
        source = row.get("untruncated_excerpt", row["presented_excerpt"])
        if not isinstance(source, str) or pipeline.content_hash(source) != row["untruncated_excerpt_hash"]:
            raise ArtifactValidationError("untruncated excerpt hash mismatch")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        grouped.setdefault(row["case_id"], []).append(row)
    for group in grouped.values():
        expected = _canonical_hash(group)
        if any(row["canonical_request_hash"] != expected for row in group):
            raise ArtifactValidationError("canonical request hash mismatch")


def is_valid_for_gating(run_dir: Path) -> bool:
    try:
        validate_run(run_dir)
    except (ArtifactValidationError, OSError, TypeError, ValueError, KeyError):
        return False
    return True

# Compatibility aliases used by callers and tests.
validate_artifact = validate_run
