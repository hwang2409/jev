"""Exact, replayable metric computations for validated memory-gate artifacts."""
from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import artifacts
import pipeline

BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 20260929
Z95 = 1.959963984540054


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson(successes: int, total: int, z: float = Z95) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [centre - half, centre + half]


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (the usual ``percentile`` method)."""
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = int(position)
    upper = min(len(values) - 1, lower + 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


def bootstrap_cases(items: Sequence[Any], evaluator: Callable[[list[Any]], float | None], *, seed: int = BOOTSTRAP_SEED, replicates: int = BOOTSTRAPS) -> dict[str, Any]:
    """Resample eligible cases and recompute the metric's ratio each time."""
    rng = random.Random(seed)
    values: list[float] = []
    dropped = 0
    items = list(items)
    for _ in range(replicates):
        sample = [items[rng.randrange(len(items))] for _ in items] if items else []
        value = evaluator(sample)
        if value is None:
            dropped += 1
        else:
            values.append(value)
    if not values:
        return {"ci": None, "null_replicates": dropped, "replicates": replicates}
    values.sort()
    return {"ci": [_percentile(values, .025), _percentile(values, .975)], "null_replicates": dropped, "replicates": replicates}


def _auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = [s for s, y in zip(scores, labels) if y]
    negatives = [s for s, y in zip(scores, labels) if not y]
    if not positives or not negatives:
        return None
    wins = sum(1 if p > n else .5 if p == n else 0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def _pr_auc(scores: Sequence[float], labels: Sequence[bool]) -> float | None:
    positives = sum(labels)
    if not positives or positives == len(labels):
        return None
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    for i in order:
        if labels[i]:
            tp += 1
        else:
            fp += 1
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _case_rows(candidates: Sequence[Mapping[str, Any]], scores: Mapping[str, float], labels: Mapping[str, str], *, tau: float, no_gate: bool = False) -> list[Mapping[str, Any]]:
    return pipeline.select_blocks(candidates, scores, tau, no_gate=no_gate)


def _normalise_cases(cases: Sequence[Mapping[str, Any]], labels: Mapping[str, str] | None = None) -> list[dict[str, Any]]:
    result = []
    for source in cases:
        candidates = list(source.get("candidates", []))
        scores = source.get("scores", {})
        if not isinstance(scores, Mapping):
            scores = {r["candidate_id"]: r["score"] for r in scores}
        local_labels = dict(labels or {})
        local_labels.update(source.get("labels", {}))
        result.append({**source, "candidates": candidates, "scores": dict(scores), "labels": local_labels})
    return result


def metrics(cases: Sequence[Mapping[str, Any]], tau: float, labels: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Compute the complete §5 table for one threshold.

    ``cases`` is intentionally a small plain mapping interface so tests can
    hand-construct cases; callers must validate the artifact before invoking
    this function.
    """
    rows = _normalise_cases(cases, labels)
    valid = [c for c in rows if c.get("valid", True)]
    for c in valid:
        pipeline.validate_scores(c["candidates"], c["scores"])
    answerable = [c for c in valid if c.get("answerable", True)]
    eligible = [c for c in answerable if any(c["labels"].get(x["candidate_id"]) == "positive" for x in c["candidates"])]
    injected = [_case_rows(c["candidates"], c["scores"], c["labels"], tau=tau) for c in valid]
    no_gate = [_case_rows(c["candidates"], c["scores"], c["labels"], tau=tau, no_gate=True) for c in eligible]

    def injected_for(c: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        return _case_rows(c["candidates"], c["scores"], c["labels"], tau=tau)

    positives = lambda blocks, c: sum(c["labels"].get(x["candidate_id"]) == "positive" for x in blocks)
    nonambiguous = lambda blocks, c: sum(c["labels"].get(x["candidate_id"]) in {"positive", "negative"} for x in blocks)
    recall = ratio(sum(bool(positives(b, c)) for c, b in zip(eligible, [injected_for(c) for c in eligible])), len(eligible))
    baseline = ratio(sum(bool(positives(b, c)) for c, b in zip(eligible, no_gate)), len(eligible))
    optimal: dict[str, set[str]] = {}
    exact = []
    for c in eligible:
        pc = [x for x in c["candidates"] if c["labels"].get(x["candidate_id"]) == "positive"]
        ps = {x["candidate_id"]: c["scores"][x["candidate_id"]] for x in pc}
        optimal[c.get("case_id", str(id(c)))] = {x["candidate_id"] for x in pipeline.select_blocks(pc, ps, tau, no_gate=True)}
        actual = {x["candidate_id"] for x in injected_for(c) if c["labels"].get(x["candidate_id"]) == "positive"}
        exact.append(actual == optimal[c.get("case_id", str(id(c)))])
    forbidden_blocks = [x for c, blocks in zip(valid, injected) for x in blocks if x.get("path") in set(c.get("forbidden_paths", []))]
    forbidden_cases = sum(any(x.get("path") in set(c.get("forbidden_paths", [])) for x in b) for c, b in zip(valid, injected))
    pair_rows = [(c["scores"][x["candidate_id"]], c["labels"].get(x["candidate_id"])) for c in valid for x in c["candidates"] if c["labels"].get(x["candidate_id"]) in {"positive", "negative"}]
    pair_scores = [x[0] for x in pair_rows]
    pair_labels = [x[1] == "positive" for x in pair_rows]
    prevalence = ratio(sum(pair_labels), len(pair_labels))
    def stratum(case: Mapping[str, Any]) -> str:
        explicit = case.get("stratum")
        return str(explicit) if explicit is not None else ("answerable" if case.get("answerable", True) else "abstain")

    strata = sorted({stratum(c) for c in valid})
    by_stratum = {
        name: {
            "rate": ratio(sum(bool(b) for c, b in zip(valid, injected) if stratum(c) == name), sum(stratum(c) == name for c in valid)),
            "cases": sum(bool(b) for c, b in zip(valid, injected) if stratum(c) == name),
            "total": sum(stratum(c) == name for c in valid),
        }
        for name in strata
    }
    result: dict[str, Any] = {
        "tau": tau,
        "any_injection_rate": ratio(sum(bool(x) for x in injected), len(valid)),
        "any_injection_rate_by_stratum": by_stratum,
        "any_injection_cases": sum(bool(x) for x in injected),
        "packet_recall": recall,
        "packet_recall_baseline": baseline,
        "retention": ratio(recall, baseline) if baseline is not None and baseline != 0 else None,
        "packet_precision": ratio(sum(positives(b, c) for c, b in zip(valid, injected)), sum(nonambiguous(b, c) for c, b in zip(valid, injected))),
        "empty_injection_cases": sum(not x for x in injected),
        "exact_packet_rate": ratio(sum(exact), len(exact)),
        "forbidden_injection_rate": ratio(forbidden_cases, len(valid)),
        "forbidden_injection_cases": forbidden_cases,
        "forbidden_injection_blocks": len(forbidden_blocks),
        "retrieval_miss_rate": ratio(sum(not any(c["labels"].get(x["candidate_id"]) == "positive" for x in c["candidates"]) for c in answerable), len(answerable)),
        "eligible_recall_cases": len(eligible),
        "ambiguous_pairs": sum(c["labels"].get(x["candidate_id"]) == "ambiguous" for c in valid for x in c["candidates"]),
        "ambiguous_cases": sum(any(c["labels"].get(x["candidate_id"]) == "ambiguous" for x in c["candidates"]) for c in valid),
        "roc_auc": _auc(pair_scores, pair_labels),
        "pr_auc": _pr_auc(pair_scores, pair_labels),
        "prevalence": prevalence,
        "brier": ratio(sum((s - y) ** 2 for s, y in zip(pair_scores, pair_labels)), len(pair_labels)),
    }
    result["bootstrap"] = {
        "any_injection_rate": bootstrap_cases(valid, lambda sample: ratio(sum(bool(injected_for(c)) for c in sample), len(sample))),
        "packet_recall": bootstrap_cases(eligible, lambda sample: ratio(sum(bool(positives(injected_for(c), c)) for c in sample), len(sample))),
        "packet_precision": bootstrap_cases(valid, lambda sample: ratio(sum(positives(injected_for(c), c) for c in sample), sum(nonambiguous(injected_for(c), c) for c in sample))),
        "exact_packet_rate": bootstrap_cases(eligible, lambda sample: ratio(sum(({x["candidate_id"] for x in injected_for(c) if c["labels"].get(x["candidate_id"]) == "positive"} == optimal[c.get("case_id", str(id(c)))]) for c in sample), len(sample))),
        "forbidden_injection_rate": bootstrap_cases(valid, lambda sample: ratio(sum(any(x.get("path") in set(c.get("forbidden_paths", [])) for x in injected_for(c)) for c in sample), len(sample))),
    }
    return result


def load_run(run_dir: Path) -> list[dict[str, Any]]:
    """Validate and convert JSONL artifacts into case records."""
    if not artifacts.is_valid_for_gating(run_dir):
        raise artifacts.ArtifactValidationError("run is not valid for gating")
    candidates = artifacts._read(run_dir / "candidates.jsonl")
    scores = {r["candidate_id"]: r for r in artifacts._read(run_dir / "scores.jsonl")}
    labels = {r["candidate_id"]: r["label"] for r in artifacts._read(run_dir / "labels.jsonl")}
    metadata = {}
    metadata_path = run_dir / "metadata.json"
    if metadata_path.exists():
        import json
        metadata = json.loads(metadata_path.read_text())
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        grouped.setdefault(row["case_id"], []).append(row)
    return [{"case_id": case_id, "candidates": rows, "scores": {x["candidate_id"]: scores[x["candidate_id"]]["score"] for x in rows}, "labels": {x["candidate_id"]: labels[x["candidate_id"]] for x in rows}, **metadata.get(case_id, {})} for case_id, rows in grouped.items()]


def observed_scores(cases: Sequence[Mapping[str, Any]]) -> list[float]:
    return sorted({float(score) for c in cases for score in c["scores"].values()})


def compute_run(run_dir: Path, tau: float) -> dict[str, Any]:
    """Validated-artifact entry point; invalid runs receive no partial metrics."""
    return metrics(load_run(run_dir), tau)
