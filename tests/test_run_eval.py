import json

import run_eval
from run_eval import auc, evaluate, summarize, top_k
from router import RouteResult


def fake_route_factory(tool, confidence, needs_tool, step_clarity):
    def fake_route(task, step, history=None):
        probs = {tool: confidence, "Other": round(1 - confidence, 4)}
        return RouteResult(
            tool=tool,
            probabilities=probs,
            confidence=confidence,
            needs_tool=needs_tool,
            step_clarity=step_clarity,
            usage={"input_tokens": 100, "output_tokens": 10},
        )
    return fake_route


def case(id, expected_tool, needs=True, vague=False):
    return {"id": id, "task": "t", "step": "s", "history": [],
            "expected_tool": expected_tool, "expected_needs_tool": needs,
            "vague": vague}


def test_top_k():
    probs = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert top_k(probs, 2) == ["A", "B"]


def test_auc_perfect_separation():
    assert auc([0.8, 0.9], [0.1, 0.2]) == 1.0


def test_auc_identical_distributions():
    assert auc([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]) == 0.5


def test_auc_mixed_case():
    assert auc([0.9, 0.6], [0.8, 0.6, 0.4]) == 0.75


def test_auc_empty_cohort():
    assert auc([], [0.5]) is None
    assert auc([0.5], []) is None


def test_summarize_metrics():
    cases_and_routes = [
        (case("c1", "Read"), fake_route_factory("Read", 0.9, 0.95, 0.9)),
        (case("c2", "Bash"), fake_route_factory("Grep", 0.4, 0.95, 0.9)),
        (case("n1", None, needs=False), fake_route_factory("Read", 0.5, 0.1, 0.7)),
        (case("v1", None, vague=True), fake_route_factory("Bash", 0.3, 0.8, 0.2)),
    ]
    results = []
    for c, fn in cases_and_routes:
        results.extend(evaluate([c], route_fn=fn))
    s = summarize(results)
    assert s["clear_cases"] == 2
    assert s["top1_accuracy"] == 0.5
    assert s["mean_confidence_correct"] == 0.9
    assert s["mean_confidence_incorrect"] == 0.4
    assert s["confusions"] == [{"id": "c2", "expected": "Bash", "chosen": "Grep"}]
    assert s["needs_tool_mean_on_tool_cases"] == 0.9
    assert s["needs_tool_mean_on_no_tool_cases"] == 0.1
    assert s["clarity_mean_on_clear"] == 0.8333
    assert s["clarity_mean_on_vague"] == 0.2
    assert s["total_input_tokens"] == 400
    assert s["total_output_tokens"] == 40
    assert s["errors"] == 0


def test_evaluate_captures_errors():
    def boom(task, step, history=None):
        raise RuntimeError("api down")
    results = evaluate([case("c1", "Read")], route_fn=boom)
    assert results[0]["error"] == "api down"
    assert summarize(results)["errors"] == 1


def test_summarize_counts_errored_clear_case_as_accuracy_miss():
    def boom(task, step, history=None):
        raise RuntimeError("api down")

    results = evaluate(
        [case("c1", "Read"), case("c2", "Bash")],
        route_fn=fake_route_factory("Read", 0.9, 0.95, 0.9),
    )
    results[1] = evaluate([case("c2", "Bash")], route_fn=boom)[0]

    summary = summarize(results)

    assert summary["clear_cases"] == 2
    assert summary["top1_accuracy"] == 0.5
    assert summary["top3_accuracy"] == 0.5
    assert summary["errors"] == 1


def test_main_writes_json_results(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_eval, "load_cases", lambda: [])
    monkeypatch.setattr(run_eval, "evaluate", lambda cases: [])

    run_eval.main()

    result_files = list((tmp_path / "results").glob("*.json"))
    assert len(result_files) == 1
    assert json.loads(result_files[0].read_text()) == {
        "summary": summarize([]),
        "results": [],
    }
