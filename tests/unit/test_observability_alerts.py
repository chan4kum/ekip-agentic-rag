"""The alert rules and dashboards only reference metrics that really exist."""

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml
from ekip_observability.metrics import EkipMetrics, normalise_metric_name

ROOT = Path(__file__).resolve().parents[2]
ALERTS = ROOT / "ops" / "prometheus" / "alerts.yaml"
DASHBOARDS = sorted((ROOT / "ops" / "grafana").glob("*.json"))

METRIC = re.compile(r"\b(ekip_[a-z0-9_]+)\b")
SELECTOR = re.compile(r"\b(ekip_[a-z0-9_]+)\s*\{([^}]*)\}")
LABEL = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=~|!~|!=|=)")


def declared_labels() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for value in vars(EkipMetrics()).values():
        name = getattr(value, "_name", None)
        labels = getattr(value, "_labelnames", None)
        if isinstance(name, str) and name.startswith("ekip_") and labels is not None:
            out[normalise_metric_name(name)] = set(labels)
    return out


def check_expression(expr: str) -> list[str]:
    known = declared_labels()
    problems = []
    for raw in METRIC.findall(expr):
        name = normalise_metric_name(raw)
        if name not in known:
            problems.append(f"unknown metric {raw}")
    for raw, body in SELECTOR.findall(expr):
        name = normalise_metric_name(raw)
        for label in LABEL.findall(body):
            if name in known and label not in known[name] | {"le"}:
                problems.append(f"{raw}: unknown label {label}")
    return problems


def alert_rules() -> list[dict[str, Any]]:
    data = yaml.safe_load(ALERTS.read_text())
    return [r for g in data["groups"] for r in g["rules"]]


def test_checker_catches_unknown_metrics_and_labels() -> None:
    assert check_expression("rate(ekip_nope_total[5m])")
    assert check_expression('rate(ekip_http_requests_total{tenant="a"}[5m])')
    assert not check_expression('rate(ekip_http_requests_total{status=~"5.."}[5m])')


@pytest.mark.parametrize("rule", alert_rules(), ids=lambda r: r["alert"])
def test_alert_uses_only_defined_metrics_and_labels(rule: dict[str, Any]) -> None:
    assert METRIC.search(rule["expr"]), "alert has no EKIP metric"
    assert check_expression(rule["expr"]) == []


@pytest.mark.parametrize("rule", alert_rules(), ids=lambda r: r["alert"])
def test_alert_is_actionable(rule: dict[str, Any]) -> None:
    assert rule["labels"]["severity"] in {"page", "ticket"}
    assert rule["annotations"]["summary"]
    assert (ROOT / rule["annotations"]["runbook"]).is_file()
    assert "for" in rule


def test_alert_names_are_unique() -> None:
    names = [r["alert"] for r in alert_rules()]
    assert len(names) == len(set(names))


def test_masterplan_slos_are_covered() -> None:
    exprs = " ".join(r["expr"] for r in alert_rules())
    assert "0.001" in exprs  # 99.9% availability budget
    assert "ekip_chat_request_duration_seconds" in exprs  # simple RAG P95 < 5s
    assert "ekip_chat_first_token_seconds" in exprs  # first token P95 < 2s
    assert "ekip_ingest_freshness_seconds" in exprs  # freshness P95 < 5 min
    assert "ekip_ingest_dlq_depth" in exprs
    assert "ekip_ingest_consumer_lag" in exprs
    assert "ekip_acl_propagation_seconds" in exprs  # permission revocation delay


def test_dashboards_exist() -> None:
    assert {p.stem for p in DASHBOARDS} == {"api", "rag", "agent", "llm", "ingestion"}


@pytest.mark.parametrize("path", DASHBOARDS, ids=lambda p: p.stem)
def test_dashboard_queries_use_defined_metrics(path: Path) -> None:
    board = json.loads(path.read_text())
    assert board["uid"].startswith("ekip-")
    assert board["panels"]
    for panel in board["panels"]:
        assert panel["targets"], panel["title"]
        for target in panel["targets"]:
            assert check_expression(target["expr"]) == [], (panel["title"], target["expr"])


def test_dashboard_ids_unique_per_board() -> None:
    for path in DASHBOARDS:
        ids = [p["id"] for p in json.loads(path.read_text())["panels"]]
        assert len(ids) == len(set(ids))
