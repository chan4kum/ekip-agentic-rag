import json
from pathlib import Path

import pytest
import yaml
from ekip_observability.evalgate import (
    Gate,
    GateConfigError,
    evaluate_gates,
    format_report,
    load_gates,
)
from ekip_retrieval.evaluation import EvalReport, Metrics
from script_loader import ROOT, load_script

GATES = (
    Gate("retrieval.leaks", "eq", 0, "retrieval"),
    Gate("retrieval.hybrid_rrf.recall", "min", 0.9, "retrieval"),
    Gate("retrieval.hybrid_rrf.ms_per_query", "max", 50, "retrieval"),
    Gate("agent.task_success", "min", 0.85, "agent"),
)


def test_improvement_and_equal_pass() -> None:
    report = evaluate_gates(
        GATES,
        {
            "retrieval.leaks": 0,
            "retrieval.hybrid_rrf.recall": 0.9,
            "retrieval.hybrid_rrf.ms_per_query": 10,
            "agent.task_success": 0.99,
        },
    )
    assert report.passed and not report.failures


def test_regression_fails() -> None:
    report = evaluate_gates(
        GATES,
        {
            "retrieval.leaks": 0,
            "retrieval.hybrid_rrf.recall": 0.89,
            "retrieval.hybrid_rrf.ms_per_query": 10,
            "agent.task_success": 0.9,
        },
    )
    assert not report.passed
    assert [f.gate.metric for f in report.failures] == ["retrieval.hybrid_rrf.recall"]


def test_any_leak_fails_even_if_quality_improved() -> None:
    report = evaluate_gates(
        GATES,
        {
            "retrieval.leaks": 1,
            "retrieval.hybrid_rrf.recall": 1.0,
            "retrieval.hybrid_rrf.ms_per_query": 1,
            "agent.task_success": 1.0,
        },
    )
    assert not report.passed


def test_missing_metric_fails_closed() -> None:
    report = evaluate_gates(GATES, {"retrieval.leaks": 0})
    assert not report.passed
    assert all(f.measured is None and f.detail == "missing measurement" for f in report.failures)


def test_nan_is_treated_as_missing() -> None:
    report = evaluate_gates(
        GATES[:2], {"retrieval.leaks": 0, "retrieval.hybrid_rrf.recall": float("nan")}
    )
    assert not report.passed


def test_suite_filter_skips_other_suites() -> None:
    report = evaluate_gates(
        GATES,
        {
            "retrieval.leaks": 0,
            "retrieval.hybrid_rrf.recall": 1,
            "retrieval.hybrid_rrf.ms_per_query": 1,
        },
        suites=["retrieval"],
    )
    assert report.passed and len(report.results) == 3


def test_report_table_lists_every_gate_and_the_verdict() -> None:
    text = format_report(evaluate_gates(GATES, {"retrieval.leaks": 2}))
    assert "retrieval.leaks" in text and "FAIL" in text and "missing measurement" in text
    assert text.rstrip().endswith("FAIL (4 gate(s))")
    passing = format_report(evaluate_gates(GATES[:1], {"retrieval.leaks": 0}))
    assert passing.rstrip().endswith("RESULT: PASS")


@pytest.mark.parametrize(
    "bad",
    [
        {"version": 2, "gates": []},
        {"version": 1, "gates": []},
        {"version": 1, "gates": [{"metric": "m", "op": "gt", "value": 1, "suite": "s"}]},
        {"version": 1, "gates": [{"metric": "m", "op": "min", "value": "1", "suite": "s"}]},
        {"version": 1, "gates": [{"metric": "m", "op": "min", "value": True, "suite": "s"}]},
        {"version": 1, "gates": [{"metric": "m", "op": "min", "value": 1}]},
        {"version": 1, "gates": [{"metric": "m", "op": "min", "value": 1, "suite": "s", "x": 1}]},
        {
            "version": 1,
            "gates": [
                {"metric": "m", "op": "min", "value": 1, "suite": "s"},
                {"metric": "m", "op": "max", "value": 1, "suite": "s"},
            ],
        },
    ],
)
def test_invalid_gate_files_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(GateConfigError):
        load_gates(bad)


def test_checked_in_gates_load_and_cover_the_required_areas() -> None:
    gates = load_gates(yaml.safe_load((ROOT / "ops" / "eval-gates.yaml").read_text()))
    by_metric = {g.metric: g for g in gates}
    assert by_metric["retrieval.leaks"].op == "eq" and by_metric["retrieval.leaks"].value == 0
    assert by_metric["agent.leaks"].value == 0
    assert by_metric["agent.injection_success"].value == 0
    assert {"retrieval", "agent", "runtime"} <= {g.suite for g in gates}
    assert any(g.metric.startswith("retrieval.") and g.op == "min" for g in gates)
    assert "agent.groundedness" in by_metric and "load.p95_ms" in by_metric
    assert "llm.cost_usd_per_request" in by_metric


def report(name: str, recall: float, leaks: tuple[str, ...] = ()) -> EvalReport:
    return EvalReport(
        name=name,
        k=5,
        overall=Metrics(10, recall, 0.8, 0.85, recall),
        by_kind={},
        leaks=leaks,
        restricted_queries=0,
        mean_latency_ms=1.5,
    )


def test_script_converts_eval_reports_to_flat_metrics() -> None:
    gate = load_script("eval_gate")
    flat = gate.retrieval_measurements(
        [report("hybrid (RRF)", 0.97), report("bm25 only", 0.9, leaks=("q1 -> p",))]
    )
    assert flat["retrieval.hybrid_rrf.recall"] == 0.97
    assert flat["retrieval.bm25_only.mrr"] == 0.8
    assert flat["retrieval.hybrid_rrf.ms_per_query"] == 1.5
    assert flat["retrieval.leaks"] == 1


def test_script_slug_is_stable_for_unknown_names() -> None:
    gate = load_script("eval_gate")
    assert gate.slug("Some New Variant!") == "some_new_variant"


def test_parse_flat_json_takes_the_last_object_and_drops_non_numbers() -> None:
    gate = load_script("eval_gate")
    out = gate.parse_flat_json('log line\n{"agent.task_success": 0.9, "note": "x", "ok": true}')
    assert out == {"agent.task_success": 0.9}


def test_offline_retrieval_suite_passes_the_checked_in_gates() -> None:
    gate = load_script("eval_gate")
    gates = load_gates(yaml.safe_load((ROOT / "ops" / "eval-gates.yaml").read_text()))
    measurements = gate.run_retrieval(5)
    assert evaluate_gates(gates, measurements, suites=["retrieval"]).passed


def test_cli_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = load_script("eval_gate")
    agent = tmp_path / "agent.json"
    agent.write_text(json.dumps({"agent.task_success": 0.1}))
    gates = tmp_path / "gates.yaml"
    gates.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "gates": [
                    {"metric": "agent.task_success", "op": "min", "value": 0.5, "suite": "agent"}
                ],
            }
        )
    )
    argv = ["eval_gate", "--gates", str(gates), "--suites", "agent", "--agent-results", str(agent)]
    monkeypatch.setattr("sys.argv", argv)
    assert gate.main() == 1
    agent.write_text(json.dumps({"agent.task_success": 0.9}))
    assert gate.main() == 0
