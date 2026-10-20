# ruff: noqa: E501  # PromQL expressions are long single strings
"""Generate the Grafana dashboards in ops/grafana/ from compact definitions.

    uv run --no-sync python scripts/gen_dashboards.py

The JSON files are committed; edit the definitions here and regenerate. tests check that every
panel query only references metrics defined in ekip_observability.metrics.
"""

import json
from pathlib import Path
from typing import Any

OUT = Path("ops/grafana")
DS = {"type": "prometheus", "uid": "${DS_PROMETHEUS}"}

Panel = tuple[str, str, list[tuple[str, str]]]  # (title, unit, [(legend, expr)])


def q(p: str, metric: str, window: str = "5m", by: str = "") -> str:
    inner = f"sum by (le{', ' + by if by else ''}) (rate({metric}_bucket[{window}]))"
    return f"histogram_quantile({p}, {inner})"


DASHBOARDS: dict[str, tuple[str, list[Panel]]] = {
    "api": (
        "EKIP / API",
        [
            ("Requests per second", "reqps", [("all", "sum(rate(ekip_http_requests_total[5m]))")]),
            (
                "5xx ratio (SLO 0.1%)",
                "percentunit",
                [
                    (
                        "5xx",
                        'sum(rate(ekip_http_requests_total{status=~"5.."}[5m])) / sum(rate(ekip_http_requests_total[5m]))',
                    )
                ],
            ),
            (
                "Latency P50 / P95 / P99",
                "s",
                [
                    ("p50", q("0.5", "ekip_http_request_duration_seconds")),
                    ("p95", q("0.95", "ekip_http_request_duration_seconds")),
                    ("p99", q("0.99", "ekip_http_request_duration_seconds")),
                ],
            ),
            ("Requests in progress", "short", [("in flight", "ekip_http_requests_in_progress")]),
            (
                "Rate-limit rejections",
                "reqps",
                [("429", "sum(rate(ekip_rate_limit_rejections_total[5m]))")],
            ),
            (
                "Chat P95 by mode (SLO 5 s)",
                "s",
                [("by mode", q("0.95", "ekip_chat_request_duration_seconds", by="mode"))],
            ),
            (
                "First token P95 (SLO 2 s)",
                "s",
                [("first token", q("0.95", "ekip_chat_first_token_seconds"))],
            ),
        ],
    ),
    "rag": (
        "EKIP / Retrieval and cache",
        [
            (
                "Retrieval latency P50 / P95",
                "s",
                [
                    ("p50", q("0.5", "ekip_retrieval_duration_seconds")),
                    ("p95", q("0.95", "ekip_retrieval_duration_seconds")),
                ],
            ),
            (
                "Hits per retrieval (median)",
                "short",
                [("median", q("0.5", "ekip_retrieval_results"))],
            ),
            (
                "Degraded retrievals by reason",
                "ops",
                [("reason", "sum by (reason) (rate(ekip_retrieval_degraded_total[5m]))")],
            ),
            (
                "Cache events by layer",
                "ops",
                [("layer/event", "sum by (layer, event) (rate(ekip_cache_events_total[5m]))")],
            ),
            (
                "Cache hit ratio",
                "percentunit",
                [
                    (
                        "hit ratio",
                        'sum(rate(ekip_cache_events_total{event="hit"}[15m])) / sum(rate(ekip_cache_events_total{event=~"hit|miss"}[15m]))',
                    )
                ],
            ),
            (
                "Blocked / stale cache entries",
                "ops",
                [
                    ("blocked", 'sum(rate(ekip_cache_events_total{event="blocked"}[5m]))'),
                    ("stale", 'sum(rate(ekip_cache_events_total{event="stale"}[5m]))'),
                ],
            ),
        ],
    ),
    "agent": (
        "EKIP / Agents",
        [
            (
                "Runs by status",
                "ops",
                [("status", "sum by (status) (rate(ekip_agent_runs_total[5m]))")],
            ),
            (
                "Graph steps (median / P95)",
                "short",
                [
                    ("p50", q("0.5", "ekip_agent_steps")),
                    ("p95", q("0.95", "ekip_agent_steps")),
                ],
            ),
            (
                "Tool calls by outcome",
                "ops",
                [
                    (
                        "tool/outcome",
                        "sum by (tool, outcome) (rate(ekip_agent_tool_calls_total[5m]))",
                    )
                ],
            ),
            (
                "Budget truncations by resource",
                "ops",
                [
                    (
                        "resource",
                        "sum by (resource) (rate(ekip_agent_budget_truncations_total[5m]))",
                    )
                ],
            ),
            (
                "Failure ratio",
                "percentunit",
                [
                    (
                        "failed",
                        'sum(rate(ekip_agent_runs_total{status="failed"}[15m])) / sum(rate(ekip_agent_runs_total[15m]))',
                    )
                ],
            ),
        ],
    ),
    "llm": (
        "EKIP / LLM",
        [
            (
                "Tokens per second",
                "short",
                [
                    (
                        "model/direction",
                        "sum by (model, direction) (rate(ekip_llm_tokens_total[5m]))",
                    )
                ],
            ),
            (
                "Provider latency P95",
                "s",
                [("model", q("0.95", "ekip_llm_request_duration_seconds", by="model"))],
            ),
            (
                "Provider errors by kind",
                "ops",
                [("model/kind", "sum by (model, kind) (rate(ekip_llm_errors_total[5m]))")],
            ),
            (
                "Estimated spend (USD / hour)",
                "currencyUSD",
                [("model", "sum by (model) (increase(ekip_llm_cost_usd_total[1h]))")],
            ),
            (
                "Error ratio",
                "percentunit",
                [
                    (
                        "errors",
                        "sum(rate(ekip_llm_errors_total[10m])) / sum(rate(ekip_llm_request_duration_seconds_count[10m]))",
                    )
                ],
            ),
        ],
    ),
    "ingestion": (
        "EKIP / Ingestion",
        [
            (
                "Documents per minute by outcome",
                "short",
                [("outcome", "sum by (outcome) (rate(ekip_ingest_documents_total[5m])) * 60")],
            ),
            (
                "Freshness P50 / P95 (SLO 5 min)",
                "s",
                [
                    ("p50", q("0.5", "ekip_ingest_freshness_seconds", "15m")),
                    ("p95", q("0.95", "ekip_ingest_freshness_seconds", "15m")),
                ],
            ),
            ("Consumer lag", "short", [("messages", "ekip_ingest_consumer_lag")]),
            ("Dead-letter queue depth", "short", [("messages", "ekip_ingest_dlq_depth")]),
            (
                "Permission propagation P95",
                "s",
                [("p95", q("0.95", "ekip_acl_propagation_seconds", "15m"))],
            ),
        ],
    ),
}


def build(uid: str, title: str, panels: list[Panel]) -> dict[str, Any]:
    out_panels = []
    for i, (name, unit, targets) in enumerate(panels):
        out_panels.append(
            {
                "id": i + 1,
                "type": "timeseries",
                "title": name,
                "datasource": DS,
                "gridPos": {"h": 8, "w": 12, "x": (i % 2) * 12, "y": (i // 2) * 8},
                "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
                "targets": [
                    {"refId": chr(65 + j), "legendFormat": legend, "expr": expr, "datasource": DS}
                    for j, (legend, expr) in enumerate(targets)
                ],
            }
        )
    return {
        "__inputs": [
            {
                "name": "DS_PROMETHEUS",
                "label": "Prometheus",
                "type": "datasource",
                "pluginId": "prometheus",
            }
        ],
        "uid": f"ekip-{uid}",
        "title": title,
        "tags": ["ekip"],
        "schemaVersion": 39,
        "version": 1,
        "editable": True,
        "time": {"from": "now-6h", "to": "now"},
        "refresh": "30s",
        "panels": out_panels,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for uid, (title, panels) in DASHBOARDS.items():
        path = OUT / f"{uid}.json"
        path.write_text(json.dumps(build(uid, title, panels), indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
