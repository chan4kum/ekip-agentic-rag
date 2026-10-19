"""Hybrid retrieval engine."""

from ekip_retrieval.context import (
    CitationReport,
    ContextAssembler,
    ContextConfig,
    TokenCounter,
    approx_tokens,
    check_citations,
)
from ekip_retrieval.evaluation import (
    EvalDataset,
    EvalReport,
    evaluate,
    format_reports,
    ingest_dataset,
    load_dataset,
)
from ekip_retrieval.fusion import reciprocal_rank_fusion
from ekip_retrieval.hybrid import (
    HybridConfig,
    HybridRetriever,
    RetrievalResult,
    RetrievalTrace,
    SearchMode,
)
from ekip_retrieval.query import MAX_QUERY_CHARS, InvalidQueryError, normalize_query
from ekip_retrieval.rerank import PassThroughReranker, TermCoverageReranker

__all__ = [
    "EvalDataset",
    "EvalReport",
    "evaluate",
    "format_reports",
    "ingest_dataset",
    "load_dataset",
    "CitationReport",
    "ContextAssembler",
    "ContextConfig",
    "TokenCounter",
    "approx_tokens",
    "check_citations",
    "MAX_QUERY_CHARS",
    "HybridConfig",
    "HybridRetriever",
    "InvalidQueryError",
    "PassThroughReranker",
    "RetrievalResult",
    "RetrievalTrace",
    "SearchMode",
    "TermCoverageReranker",
    "normalize_query",
    "reciprocal_rank_fusion",
]
