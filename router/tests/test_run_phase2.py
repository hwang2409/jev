from router import RouteResult
from run_phase2 import (
    brier_score,
    evaluate_cases,
    format_report,
    run_curve,
    summarize_confidence_bins,
    summarize_full,
    summarize_routing,
)


def case(id, expected_tool):
    return {
        "id": id,
        "task": "task",
        "step": "step",
        "history": [],
        "expected_tool": expected_tool,
        "expected_needs_tool": True,
        "vague": False,
    }


def routed(tool, confidence, probabilities=None):
    return RouteResult(
        tool=tool,
        probabilities=probabilities or {tool: confidence},
        confidence=confidence,
        needs_tool=1.0,
        step_clarity=1.0,
        usage={"input_tokens": 10, "output_tokens": 2},
    )


def test_curve_passes_the_matching_subset_to_route():
    seen_sizes = []

    def fake_route(task, step, history=None, catalog=None):
        seen_sizes.append(len(catalog))
        return routed("files_read_document", 0.9)

    data = run_curve([case("curve-1", "files_read_document")], fake_route)

    assert seen_sizes == [15, 30, 60, 120, 180, 250]
    assert list(data) == [15, 30, 60, 120, 180, 250]


def test_confidence_bin_edges_are_left_inclusive():
    results = [
        {"id": "a", "confidence": 0.5, "tool": "x", "expected_tool": "x"},
        {"id": "b", "confidence": 0.95, "tool": "x", "expected_tool": "y"},
    ]

    bins = summarize_confidence_bins(results)

    assert bins["[0.5-0.8)"]["cases"] == 1
    assert bins["[0.95-1.0]"]["cases"] == 1
    assert bins["[0.5-0.8)"]["accuracy"] == 1.0
    assert bins["[0.95-1.0]"]["accuracy"] == 0.0


def test_brier_uses_probability_of_chosen_tool():
    results = [
        {"tool": "a", "expected_tool": "a", "chosen_probability": 0.2},
        {"tool": "b", "expected_tool": "a", "chosen_probability": 0.7},
        {"tool": "c", "expected_tool": "c", "chosen_probability": 0.9},
    ]

    assert brier_score(results) == 0.38


def test_errors_are_misses_in_curve_accuracy():
    results = [
        {"id": "ok", "tool": "a", "expected_tool": "a", "probabilities": {"a": 1.0},
         "confidence": 1.0, "usage": {"input_tokens": 1, "output_tokens": 2}},
        {"id": "error", "expected_tool": "a", "error": "api down"},
    ]

    summary = summarize_routing(results)

    assert summary["cases"] == 2
    assert summary["errors"] == 1
    assert summary["top1_accuracy"] == 0.5
    assert summary["top3_accuracy"] == 0.5


def test_curve_report_shows_wrong_errors_and_total_misses():
    def fake_route(task, step, history=None, catalog=None):
        if task == "error":
            raise RuntimeError("api down")
        return routed("files_read_document", 0.9)

    error_case = case("error", "files_read_document")
    error_case["task"] = "error"
    data = run_curve([case("ok", "files_read_document"), error_case], fake_route)

    summary = data[15]["summary"]
    assert summary["wrong"] == 0
    assert summary["errors"] == 1
    assert summary["total_misses"] == 1

    report = format_report({15: data[15]}, None)
    assert "wrong  errors  total_misses" in report
    curve_line = next(line for line in report.splitlines() if line.lstrip().startswith("15"))
    fields = curve_line.split()
    assert fields[6:9] == ["0", "1", "1"]


def test_errors_are_misses_in_full_accuracy():
    def fake_route(task, step, history=None, catalog=None):
        if task == "error":
            raise RuntimeError("api down")
        return routed("files_read_document", 0.9)

    results = [case("cov-files_read_document", "files_read_document"),
               case("hard-1", "files_write_document")]
    results[1]["task"] = "error"
    summary = summarize_full(
        evaluate_cases(results, {"files_read_document": "read"}, fake_route)
    )

    assert summary["top1_accuracy"] == 0.5
    assert summary["errors"] == 1
    assert summary["hard"]["top1_accuracy"] == 0.0


def test_full_summary_splits_coverage_and_hard_cases():
    results = [
        {"id": "cov-a", "tool": "a", "expected_tool": "a", "probabilities": {"a": 1.0},
         "confidence": 1.0, "chosen_probability": 1.0,
         "usage": {"input_tokens": 1, "output_tokens": 2}},
        {"id": "hard-1", "tool": "b", "expected_tool": "a", "probabilities": {"b": 0.6},
         "confidence": 0.6, "chosen_probability": 0.6,
         "usage": {"input_tokens": 1, "output_tokens": 2}},
        {"id": "hard-2", "expected_tool": "a", "error": "api down"},
    ]

    summary = summarize_full(results)

    assert summary["top1_accuracy"] == 0.3333
    assert summary["coverage"]["top1_accuracy"] == 1.0
    assert summary["hard"]["top1_accuracy"] == 0.0
    assert summary["hard"]["errors"] == 1
