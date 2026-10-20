"""Observability building blocks: logging, request context, tracing, metrics, LLMOps helpers."""

from ekip_observability.asgi import MetricsMiddleware, install_metrics
from ekip_observability.canary import (
    CanaryDecision,
    RateSample,
    Verdict,
    combine,
    compare_latency_p95,
    compare_rates,
)
from ekip_observability.capacity import CorpusAssumptions, backfill_hours, replicated_vector_gb
from ekip_observability.evalgate import (
    Gate,
    GateConfigError,
    GateReport,
    evaluate_gates,
    format_report,
    load_gates,
)
from ekip_observability.llm_tracing import (
    GenAI,
    LLMTracer,
    NoopLLMTracer,
    OtelLLMTracer,
    OtlpExporterConfig,
    add_otlp_export,
    langfuse_otlp_config,
)
from ekip_observability.metrics import CostEstimator, EkipMetrics, normalise_metric_name
from ekip_observability.registry import check_prompts, prompt_entry, sha256_text
from ekip_observability.stats import percentile

__all__ = [
    "CanaryDecision",
    "CorpusAssumptions",
    "CostEstimator",
    "EkipMetrics",
    "Gate",
    "GateConfigError",
    "GateReport",
    "GenAI",
    "LLMTracer",
    "MetricsMiddleware",
    "NoopLLMTracer",
    "OtelLLMTracer",
    "OtlpExporterConfig",
    "RateSample",
    "Verdict",
    "add_otlp_export",
    "backfill_hours",
    "check_prompts",
    "combine",
    "compare_latency_p95",
    "compare_rates",
    "evaluate_gates",
    "format_report",
    "install_metrics",
    "langfuse_otlp_config",
    "load_gates",
    "normalise_metric_name",
    "percentile",
    "prompt_entry",
    "replicated_vector_gb",
    "sha256_text",
]
