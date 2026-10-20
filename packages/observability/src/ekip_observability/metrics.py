"""Prometheus metrics for the whole platform (one registry object, injected; no globals).

Cardinality rule (enforced by tests): labels hold small, fixed vocabularies only - route
TEMPLATES, status codes, outcome names, model names. NEVER a tenant, user, question, document or
conversation id: those belong in traces and logs, not in time series. `safe_label` maps anything
unexpected to "other" so a bug cannot create unbounded series.
"""

import re
from collections.abc import Mapping

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

# Label NAMES that must never exist (they would be per-user / per-document series).
FORBIDDEN_LABEL_NAMES = frozenset(
    {
        "tenant",
        "tenant_id",
        "user",
        "user_id",
        "subject",
        "sub",
        "email",
        "question",
        "query",
        "prompt",
        "document",
        "document_id",
        "chunk_id",
        "conversation_id",
        "request_id",
        "trace_id",
        "session",
    }
)

_SAFE = re.compile(r"^[A-Za-z0-9_.:/{}\-]{1,48}$")


def safe_label(value: object, *, default: str = "other") -> str:
    """Return `value` as a label only if it looks like a short identifier, else `default`."""
    text = str(value)
    return text if _SAFE.match(text) else default


def bounded_label(value: object, allowed: frozenset[str], *, default: str = "other") -> str:
    text = str(value)
    return text if text in allowed else default


_LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60)
_FAST_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
_FRESHNESS_BUCKETS = (1, 5, 15, 30, 60, 120, 300, 600, 1800, 3600, 14400, 86400)

CHAT_MODES = frozenset({"simple", "multi_step"})
AGENT_STATUSES = frozenset(
    {"answered", "insufficient_evidence", "refused", "failed", "approval_pending", "running"}
)
CACHE_LAYERS = frozenset({"response", "retrieval", "embedding", "semantic"})
CACHE_EVENTS = ("hit", "miss", "stale", "blocked")
INGEST_OUTCOMES = frozenset(
    {"indexed", "skipped", "failed", "deleted", "acl_updated", "dead_lettered"}
)
LLM_OPERATIONS = frozenset({"chat", "embeddings", "rerank"})
LLM_ERROR_KINDS = frozenset(
    {"timeout", "rate_limited", "auth", "server", "client", "invalid", "other"}
)
BUDGET_RESOURCES = frozenset({"steps", "retrievals", "tool_calls", "model_calls", "tokens", "time"})
TOOL_OUTCOMES = frozenset({"ok", "denied", "error", "timeout", "circuit_open", "approval_required"})


class CostEstimator:
    """USD estimate from a static per-million-token price table (the hook for cost analytics).

    Prices are configuration, not truth: provider price lists change. Unknown model => 0.0 so the
    metric never lies with a made-up number (the token counters still show usage).
    """

    def __init__(self, prices_per_million: Mapping[str, tuple[float, float]] | None = None) -> None:
        self._prices = dict(prices_per_million or {})

    def estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        price = self._prices.get(model)
        if price is None:
            return 0.0
        return (input_tokens * price[0] + output_tokens * price[1]) / 1_000_000


class EkipMetrics:
    """All platform metrics, registered on one private registry."""

    def __init__(
        self, registry: CollectorRegistry | None = None, *, cost: CostEstimator | None = None
    ) -> None:
        self.registry = registry or CollectorRegistry()
        self._cost = cost or CostEstimator()
        r = self.registry
        # ---- HTTP / API ----
        self.http_requests = Counter(
            "ekip_http_requests_total", "HTTP requests", ("method", "route", "status"), registry=r
        )
        self.http_duration = Histogram(
            "ekip_http_request_duration_seconds",
            "HTTP request duration (until the response has been fully sent)",
            ("method", "route"),
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.http_in_progress = Gauge(
            "ekip_http_requests_in_progress", "Requests currently being served", registry=r
        )
        self.rate_limit_rejections = Counter(
            "ekip_rate_limit_rejections_total", "Requests rejected by the rate limiter", registry=r
        )
        self.chat_duration = Histogram(
            "ekip_chat_request_duration_seconds",
            "End-to-end chat duration",
            ("mode",),
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.chat_first_token = Histogram(
            "ekip_chat_first_token_seconds",
            "Time from request start to the first streamed answer chunk",
            ("mode",),
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        # ---- RAG ----
        self.retrieval_duration = Histogram(
            "ekip_retrieval_duration_seconds",
            "Retrieval latency (query to ranked, authorized hits)",
            ("mode",),
            buckets=_FAST_BUCKETS,
            registry=r,
        )
        self.retrieval_results = Histogram(
            "ekip_retrieval_results",
            "Hits returned per retrieval",
            buckets=(0, 1, 2, 3, 5, 8, 13, 21, 34),
            registry=r,
        )
        self.retrieval_degraded = Counter(
            "ekip_retrieval_degraded_total",
            "Retrievals that fell back to a weaker mode",
            ("reason",),
            registry=r,
        )
        # ---- Agent ----
        self.agent_runs = Counter(
            "ekip_agent_runs_total", "Agent runs by final status", ("status",), registry=r
        )
        self.agent_steps = Histogram(
            "ekip_agent_steps",
            "Graph steps per agent run",
            buckets=(1, 2, 3, 4, 6, 8, 10, 12, 16),
            registry=r,
        )
        self.agent_tool_calls = Counter(
            "ekip_agent_tool_calls_total",
            "Tool calls made by agents",
            ("tool", "outcome"),
            registry=r,
        )
        self.agent_budget_truncations = Counter(
            "ekip_agent_budget_truncations_total",
            "Runs cut short by an execution budget",
            ("resource",),
            registry=r,
        )
        # ---- LLM ----
        self.llm_tokens = Counter(
            "ekip_llm_tokens_total", "LLM tokens", ("model", "direction"), registry=r
        )
        self.llm_duration = Histogram(
            "ekip_llm_request_duration_seconds",
            "Provider call duration",
            ("model", "operation"),
            buckets=_LATENCY_BUCKETS,
            registry=r,
        )
        self.llm_errors = Counter(
            "ekip_llm_errors_total", "Provider call failures", ("model", "kind"), registry=r
        )
        self.llm_cost = Counter(
            "ekip_llm_cost_usd_total",
            "Estimated spend (static price table)",
            ("model",),
            registry=r,
        )
        # ---- Ingestion ----
        self.ingest_documents = Counter(
            "ekip_ingest_documents_total", "Documents processed", ("outcome",), registry=r
        )
        self.ingest_lag = Histogram(
            "ekip_ingest_freshness_seconds",
            "Source change to searchable (document freshness)",
            buckets=_FRESHNESS_BUCKETS,
            registry=r,
        )
        self.acl_propagation = Histogram(
            "ekip_acl_propagation_seconds",
            "Permission change at the source to enforced in the index",
            buckets=_FRESHNESS_BUCKETS,
            registry=r,
        )
        self.ingest_dlq_depth = Gauge(
            "ekip_ingest_dlq_depth", "Messages waiting in the dead-letter topic", registry=r
        )
        self.ingest_consumer_lag = Gauge(
            "ekip_ingest_consumer_lag", "Messages the change consumer is behind", registry=r
        )
        # ---- Cache ----
        self.cache_events = Counter(
            "ekip_cache_events_total", "Cache events", ("layer", "event"), registry=r
        )

    # ---- HTTP -------------------------------------------------------------------------------
    def observe_http(self, *, method: str, route: str, status: int, seconds: float) -> None:
        method = bounded_label(method, frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}))
        route = safe_label(route, default="unmatched")
        self.http_requests.labels(method, route, str(status)).inc()
        self.http_duration.labels(method, route).observe(seconds)

    def observe_chat(self, *, mode: str, seconds: float, first_token_seconds: float | None) -> None:
        mode = bounded_label(mode, CHAT_MODES)
        self.chat_duration.labels(mode).observe(seconds)
        if first_token_seconds is not None:
            self.chat_first_token.labels(mode).observe(first_token_seconds)

    # ---- RAG --------------------------------------------------------------------------------
    def observe_retrieval(self, *, mode: str, seconds: float, results: int) -> None:
        self.retrieval_duration.labels(safe_label(mode)).observe(seconds)
        self.retrieval_results.observe(results)

    def retrieval_degraded_event(self, reason: str) -> None:
        self.retrieval_degraded.labels(safe_label(reason)).inc()

    # ---- Agent ------------------------------------------------------------------------------
    def observe_agent_run(self, *, status: str, steps: int, truncated: str | None = None) -> None:
        self.agent_runs.labels(bounded_label(status, AGENT_STATUSES)).inc()
        self.agent_steps.observe(steps)
        if truncated is not None:
            self.agent_budget_truncations.labels(bounded_label(truncated, BUDGET_RESOURCES)).inc()

    def tool_call(self, *, tool: str, outcome: str) -> None:
        self.agent_tool_calls.labels(safe_label(tool), bounded_label(outcome, TOOL_OUTCOMES)).inc()

    # ---- LLM --------------------------------------------------------------------------------
    def observe_llm(
        self,
        *,
        model: str,
        operation: str,
        seconds: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        error_kind: str | None = None,
    ) -> float:
        """Record one provider call; returns the estimated cost in USD."""
        model = safe_label(model)
        self.llm_duration.labels(model, bounded_label(operation, LLM_OPERATIONS)).observe(seconds)
        if error_kind is not None:
            self.llm_errors.labels(model, bounded_label(error_kind, LLM_ERROR_KINDS)).inc()
        if input_tokens:
            self.llm_tokens.labels(model, "input").inc(input_tokens)
        if output_tokens:
            self.llm_tokens.labels(model, "output").inc(output_tokens)
        usd = self._cost.estimate(model, input_tokens, output_tokens)
        if usd:
            self.llm_cost.labels(model).inc(usd)
        return usd

    # ---- Ingestion --------------------------------------------------------------------------
    def ingest_outcome(self, outcome: str, *, freshness_seconds: float | None = None) -> None:
        self.ingest_documents.labels(bounded_label(outcome, INGEST_OUTCOMES)).inc()
        if freshness_seconds is not None:
            self.ingest_lag.observe(freshness_seconds)

    def acl_propagated(self, seconds: float) -> None:
        self.acl_propagation.observe(seconds)

    def set_dlq_depth(self, depth: int) -> None:
        self.ingest_dlq_depth.set(depth)

    def set_consumer_lag(self, lag: int) -> None:
        self.ingest_consumer_lag.set(lag)

    # ---- Cache (shape-compatible with ekip_caching's CacheMetrics Protocol) ------------------
    def _cache(self, layer: str, event: str) -> None:
        self.cache_events.labels(bounded_label(layer, CACHE_LAYERS), event).inc()

    def hit(self, layer: str) -> None:
        self._cache(layer, "hit")

    def miss(self, layer: str) -> None:
        self._cache(layer, "miss")

    def stale(self, layer: str) -> None:
        self._cache(layer, "stale")

    def blocked(self, layer: str) -> None:
        self._cache(layer, "blocked")

    # ---- Introspection (used by tests and the alert/dashboard validators) --------------------
    def family_names(self) -> frozenset[str]:
        """Metric family names as Prometheus exposes them without type suffixes."""
        return frozenset(normalise_metric_name(f.name) for f in self.registry.collect())


_SUFFIXES = ("_bucket", "_count", "_sum", "_total", "_created")


def normalise_metric_name(name: str) -> str:
    """Strip the suffixes Prometheus adds, so `x_total`, `x_bucket` and `x` compare equal."""
    changed = True
    while changed:
        changed = False
        for suffix in _SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
    return name
