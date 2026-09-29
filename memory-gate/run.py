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
import hashlib
import json
import math
import random
import statistics
import sys
from collections.abc import Iterable, Mapping, Sequence
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
REQUIRED_PROVENANCE = {
    "case_set_fingerprint", "corpus_fingerprint", "pausanias_revision",
    "retrieval_config", "rank", "untruncated_excerpt_hash",
    "presented_excerpt", "canonical_request_hash", "production_builder_hash",
    "configured_model_id", "served_model_id", "harness_revision", "jm_revision",
}

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def fingerprint(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n")


def validate_record(record: Mapping[str, Any], kind: str) -> None:
    if not isinstance(record, Mapping) or not isinstance(record.get("case_id"), str):
        raise TypeError(f"{kind}: case_id is required")
    if kind in {"candidate", "score"}:
        missing = REQUIRED_PROVENANCE - record.keys()
        if missing:
            raise ValueError(f"{kind} {record['case_id']}: missing provenance {sorted(missing)}")
        if not isinstance(record["presented_excerpt"], str) or not isinstance(record["untruncated_excerpt_hash"], str):
            raise ValueError(f"{kind} {record['case_id']}: excerpt/hash types")
    if kind == "candidate":
        for field in ("candidate_id", "query", "path", "heading", "rank"):
            if field not in record:
                raise ValueError(f"candidate {record['case_id']}: missing {field}")
        if not isinstance(record["heading"], list) or not all(isinstance(x, str) for x in record["heading"]):
            raise ValueError("candidate heading must be a string list")
    elif kind == "score":
        score = record.get("score")
        if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float)) or not 0 <= score <= 1):
            raise ValueError(f"score {record['case_id']}: invalid score")
        if not isinstance(record.get("coverage", False), bool):
            raise ValueError("coverage must be boolean")
    elif kind == "label":
        if record.get("label") not in {"positive", "negative", "ambiguous"}:
            raise ValueError("label must be positive, negative, or ambiguous")



def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return [centre - half, centre + half]


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def case_metric(rows: list[dict[str, Any]], predicate: Any, eligible: list[dict[str, Any]] | None = None) -> tuple[float | None, int]:
    sample = eligible if eligible is not None else rows
    return ratio(sum(bool(predicate(r)) for r in sample), len(sample)), len(sample)


def bootstrap(values: list[float | None], seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    rng = random.Random(seed)
    usable = [v for v in values if v is not None]
    if not usable:
        return {"ci": None, "null_replicates": BOOTSTRAPS}
    reps: list[float] = []
    dropped = 0
    for _ in range(BOOTSTRAPS):
        draw = [rng.choice(usable) for _ in usable]
        value = statistics.mean(draw) if draw else None
        if value is None:
            dropped += 1
        else:
            reps.append(value)
    reps.sort()
    return {"ci": [reps[int(.025 * (len(reps)-1))], reps[int(.975 * (len(reps)-1))]], "null_replicates": dropped}


def legacy_metrics(cases: list[dict[str, Any]], tau: float, labels: Mapping[str, str]) -> dict[str, Any]:
    valid = [c for c in cases if c.get("valid", True)]
    answerable = [c for c in valid if c.get("answerable", True)]
    eligible = [c for c in answerable if any(labels.get(x["candidate_id"]) == "positive" for x in c["candidates"])]
    validated_scores = {
        id(c): validate_scores(c["candidates"], c["scores"])
        for c in valid
    }
    injected = [select_blocks(c["candidates"], validated_scores[id(c)], tau) for c in valid]
    positive_injected = sum(1 for blocks in injected for x in blocks if labels.get(x["candidate_id"]) == "positive")
    nonambiguous = sum(1 for blocks in injected for x in blocks if labels.get(x["candidate_id"]) in {"positive", "negative"})
    recall = ratio(sum(any(labels.get(x["candidate_id"]) == "positive" for x in blocks) for blocks in [select_blocks(c["candidates"], validated_scores[id(c)], tau) for c in eligible]), len(eligible))
    no_gate = [select_blocks(c["candidates"], validated_scores[id(c)], tau, no_gate=True) for c in eligible]
    baseline = ratio(sum(any(labels.get(x["candidate_id"]) == "positive" for x in b) for b in no_gate), len(eligible))
    forbidden = sum(any(x["path"] in c.get("forbidden_paths", []) for x in b) for c, b in zip(valid, injected))
    return {
        "tau": tau, "any_injection_rate": ratio(sum(bool(b) for b in injected), len(valid)),
        "packet_recall": recall, "packet_recall_baseline": baseline,
        "retention": ratio(recall, baseline) if baseline else None,
        "packet_precision": ratio(positive_injected, nonambiguous),
        "empty_injection_cases": sum(not b for b in injected),
        "forbidden_injection_rate": ratio(forbidden, len(valid)),
        "forbidden_injection_blocks": sum(1 for c,b in zip(valid,injected) for x in b if x["path"] in c.get("forbidden_paths", [])),
        "retrieval_miss_rate": ratio(sum(not any(labels.get(x["candidate_id"]) == "positive" for x in c["candidates"]) for c in answerable), len(answerable)),
        "eligible_recall_cases": len(eligible), "ambiguous_pairs": sum(labels.get(x["candidate_id"]) == "ambiguous" for c in valid for x in c["candidates"]),
    }


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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("candidates", "score", "label-template", "lock", "report", "posthoc-safety-curve"):
        command_parser = sub.add_parser(name)
        if name == "score":
            command_parser.add_argument("--remote", default="origin", help="Git remote for witness verification (default: origin)")
    parser.epilog = "Safety verification fetches the configured remote with pruning before accepting a witness."
    args, unknown = parser.parse_known_args(argv)
    if args.command == "label-template":
        run_dir = Path(unknown[unknown.index("--candidates") + 1]) if "--candidates" in unknown else Path("memory-gate/runs/fixture-dev")
        for row in read_jsonl(run_dir / "candidates.jsonl"):
            output = dict(row)
            output["label"] = None
            print(json.dumps(output, ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "lock":
        p = Path(unknown[unknown.index("--run") + 1]) if "--run" in unknown else Path("memory-gate/runs/fixture-dev")
        tau = float(unknown[unknown.index("--tau") + 1]) if "--tau" in unknown else TAU_REFERENCE
        lock_witness(p / "LOCK.json", [p / "candidates.jsonl", p / "scores.jsonl", p / "labels.jsonl", p / "report.md"], tau)
        return 0
    if args.command == "score" and "--lane" in unknown and unknown[unknown.index("--lane") + 1] == "safety":
        if "--witness" not in unknown:
            raise SystemExit("safety scoring requires --witness <commit>")
        run = Path(unknown[unknown.index("--run") + 1])
        remote = args.remote
        verify_witness(run / "LOCK.json", unknown[unknown.index("--witness") + 1], remote=remote, safety_outputs=[run / "safety.json", run / "safety-scores.jsonl"])
        raise SystemExit("phase C not implemented: safety scoring is deferred")
    if args.command == "report":
        run_dir = Path(unknown[unknown.index("--run") + 1])
        # Refuse before reading partial streams: validity is all-or-nothing.
        validate_artifact(run_dir)
        cases = metrics_module.load_run(run_dir)
        result = sweep_module.sweep(cases)
        lines = ["RESULTS — memory-gate metrics", "", "Thresholds: " + ", ".join(f"{x:.6g}" for x in result["thresholds"]), "", "tau | any-injection | recall | precision | exact-packet | forbidden | retention | ROC-AUC | PR-AUC | Brier", "--- | --- | --- | --- | --- | --- | --- | --- | --- | ---"]
        for row in result["table"]:
            def fmt(value):
                return "null" if value is None else f"{value:.6g}"
            lines.append(" | ".join([fmt(row[k]) for k in ("tau", "any_injection_rate", "packet_recall", "packet_precision", "exact_packet_rate", "forbidden_injection_rate", "retention", "roc_auc", "pr_auc", "brier")]))
        lines += ["", "Per-stratum any-injection (at each tau):"]
        for row in result["table"]:
            strata = ", ".join(f"{name}={value['cases']}/{value['total']} ({value['rate']!r})" for name, value in row["any_injection_rate_by_stratum"].items())
            lines.append(f"tau={row['tau']:.6g}: {strata}")
        lines += ["", "Pareto frontier (abstain any-injection, packet recall):"]
        lines += [f"tau={r['tau']:.6g} any={r['any_injection_rate']!r} recall={r['packet_recall']!r}" for r in result["pareto"]]
        lines += ["", "Selection: " + json.dumps(result["selection"], sort_keys=True)]
        text = "\\n".join(lines) + "\\n"
        (run_dir / "results.txt").write_text(text)
        print(text, end="")
        return 0
    if args.command == "posthoc-safety-curve":
        run_dir = Path(unknown[unknown.index("--run") + 1])
        validate_artifact(run_dir)
        print((run_dir / "report.md").read_text())
        return 0
    if args.command == "candidates":
        raise SystemExit("candidate generation requires the local pausanias checkout and model bundle; no network fallback")
    if args.command == "score":
        raise SystemExit("score requires an offline response cache; use the homelab runner")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
