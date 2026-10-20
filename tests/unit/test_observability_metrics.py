import re

from ekip_observability.metrics import (
    FORBIDDEN_LABEL_NAMES,
    CostEstimator,
    EkipMetrics,
    normalise_metric_name,
    safe_label,
)
from prometheus_client import generate_latest


def exposition(m: EkipMetrics) -> str:
    return generate_latest(m.registry).decode()


def sample(m: EkipMetrics, name: str, **labels: str) -> float:
    value = m.registry.get_sample_value(name, labels)
    return 0.0 if value is None else value


def test_no_metric_has_a_per_user_or_per_document_label() -> None:
    m = EkipMetrics()
    for family in m.registry.collect():
        for s in family.samples:
            assert FORBIDDEN_LABEL_NAMES.isdisjoint(s.labels), (family.name, s.labels)


def test_every_declared_label_name_is_not_forbidden() -> None:
    m = EkipMetrics()
    # Touch every labelled metric once so label names show up in the exposition.
    m.observe_http(method="GET", route="/x", status=200, seconds=0.1)
    m.observe_chat(mode="simple", seconds=1, first_token_seconds=0.5)
    m.observe_retrieval(mode="hybrid", seconds=0.1, results=3)
    m.retrieval_degraded_event("vector_down")
    m.observe_agent_run(status="answered", steps=3, truncated="steps")
    m.tool_call(tool="search_documents", outcome="ok")
    m.observe_llm(model="m", operation="chat", seconds=1, input_tokens=1, output_tokens=1)
    m.ingest_outcome("indexed", freshness_seconds=1)
    m.hit("response")
    names = set(re.findall(r'(\w+)="', exposition(m)))
    assert names.isdisjoint(FORBIDDEN_LABEL_NAMES)


def test_unbounded_values_collapse_instead_of_creating_series() -> None:
    m = EkipMetrics()
    for i in range(50):
        m.observe_agent_run(status=f"made-up-{i}", steps=1)
        m.hit(f"layer-{i}")
        m.tool_call(tool="search", outcome=f"weird {i}")
        m.observe_http(method="BREW", route=f"/user/{'x' * 100}{i}", status=200, seconds=0.01)
    text = exposition(m)
    assert 'status="other"' in text
    assert 'layer="other"' in text
    assert 'outcome="other"' in text
    assert 'route="unmatched"' in text  # over-long / odd route values never become labels
    assert text.count("ekip_agent_runs_total{") <= 2


def test_safe_label_rejects_questions_and_emails() -> None:
    assert safe_label("/api/v1/chat") == "/api/v1/chat"
    assert safe_label("what is the salary of jane@corp.com?") == "other"
    assert safe_label("a" * 200) == "other"


def test_llm_metrics_and_cost_hook() -> None:
    m = EkipMetrics(cost=CostEstimator({"gpt-x": (1.0, 4.0)}))
    usd = m.observe_llm(
        model="gpt-x", operation="chat", seconds=0.5, input_tokens=1_000_000, output_tokens=500_000
    )
    assert usd == 3.0
    assert sample(m, "ekip_llm_tokens_total", model="gpt-x", direction="input") == 1_000_000
    assert sample(m, "ekip_llm_tokens_total", model="gpt-x", direction="output") == 500_000
    assert sample(m, "ekip_llm_cost_usd_total", model="gpt-x") == 3.0


def test_unknown_model_costs_zero_not_a_guess() -> None:
    m = EkipMetrics()
    assert m.observe_llm(model="unknown", operation="chat", seconds=0.1, input_tokens=10) == 0.0
    assert "ekip_llm_cost_usd_total{" not in exposition(m)


def test_llm_error_counted_by_kind() -> None:
    m = EkipMetrics()
    m.observe_llm(model="m", operation="chat", seconds=1, error_kind="timeout")
    assert sample(m, "ekip_llm_errors_total", model="m", kind="timeout") == 1


def test_cache_methods_match_the_cache_metrics_shape() -> None:
    m = EkipMetrics()
    m.hit("response")
    m.miss("response")
    m.stale("retrieval")
    m.blocked("response")
    assert sample(m, "ekip_cache_events_total", layer="response", event="hit") == 1
    assert sample(m, "ekip_cache_events_total", layer="response", event="miss") == 1
    assert sample(m, "ekip_cache_events_total", layer="retrieval", event="stale") == 1
    assert sample(m, "ekip_cache_events_total", layer="response", event="blocked") == 1


def test_agent_and_ingestion_hooks() -> None:
    m = EkipMetrics()
    m.observe_agent_run(status="answered", steps=4, truncated="retrievals")
    m.set_dlq_depth(7)
    m.set_consumer_lag(120)
    m.ingest_outcome("dead_lettered")
    m.acl_propagated(3.0)
    assert sample(m, "ekip_agent_budget_truncations_total", resource="retrievals") == 1
    assert sample(m, "ekip_ingest_dlq_depth") == 7
    assert sample(m, "ekip_ingest_consumer_lag") == 120
    assert sample(m, "ekip_ingest_documents_total", outcome="dead_lettered") == 1
    assert sample(m, "ekip_acl_propagation_seconds_count") == 1


def test_registries_are_independent() -> None:
    a, b = EkipMetrics(), EkipMetrics()
    a.hit("response")
    assert sample(b, "ekip_cache_events_total", layer="response", event="hit") == 0


def test_normalise_metric_name() -> None:
    assert normalise_metric_name("ekip_x_seconds_bucket") == "ekip_x_seconds"
    assert normalise_metric_name("ekip_x_total") == "ekip_x"
    assert normalise_metric_name("ekip_x_seconds_count") == "ekip_x_seconds"
    assert "ekip_http_requests" in EkipMetrics().family_names()
