"""Compare retrieval variants on the versioned dataset.

    uv run python scripts/eval_retrieval.py            # offline: hashing embedder, in-memory index
    uv run python scripts/eval_retrieval.py --live     # real OpenSearch + configured embedder
    uv run python scripts/eval_retrieval.py --dataset other.json --k 5

Exit code is 1 if ANY variant leaked a passage the caller may not read (that is a security bug, not
a quality number). The --live run needs the Compose stack (OpenSearch) and, for Gemini embeddings,
GOOGLE_API_KEY in .env. It uses its own throw-away index and deletes it afterwards.
"""

import argparse
import asyncio
import sys
from pathlib import Path

import httpx
from ekip_config.settings import load_settings
from ekip_contracts.embedding import Embedder
from ekip_llm_gateway import GeminiEmbedder, HashingEmbedder
from ekip_retrieval import (
    HybridConfig,
    HybridRetriever,
    SearchMode,
    TermCoverageReranker,
    evaluate,
    format_reports,
    ingest_dataset,
    load_dataset,
)
from ekip_retrieval.evaluation import EvalReport
from ekip_search.memory import InMemoryChunkIndex
from ekip_search.opensearch import OpenSearchChunkIndex

DEFAULT_DATASET = Path("evaluations/datasets/retrieval_v1.json")


async def run(dataset_path: Path, *, live: bool, k: int) -> list[EvalReport]:
    dataset = load_dataset(dataset_path)
    settings = load_settings()
    embedder: Embedder
    if live and settings.embedding_provider == "gemini":
        key = settings.google_api_key or settings.gemini_api_key
        if key is None:
            raise SystemExit("GOOGLE_API_KEY is not set (needed for --live with Gemini)")
        embedder = GeminiEmbedder(
            key.get_secret_value(),
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
    else:
        embedder = HashingEmbedder(settings.embedding_dimensions if live else 64)
    os_index: OpenSearchChunkIndex | None = None
    index: InMemoryChunkIndex | OpenSearchChunkIndex
    cleanup: list[str] = []
    if live:
        name = f"ekip-eval-{abs(hash(dataset.version)) % 10_000}"
        os_index = OpenSearchChunkIndex(
            str(settings.opensearch_url),
            model_version=embedder.model_version,
            dimensions=embedder.dimensions,
            index_name=name,
        )
        cleanup.append(name)
        index = os_index
    else:
        index = InMemoryChunkIndex(
            model_version=embedder.model_version, dimensions=embedder.dimensions
        )
    try:
        await index.ensure_ready()
        chunk_to_key = await ingest_dataset(index, embedder, dataset)
        if live:  # make freshly written chunks searchable
            async with httpx.AsyncClient(trust_env=True) as c:
                await c.post(f"{settings.opensearch_url}/{cleanup[0]}/_refresh")
        variants = {
            "bm25 only": (SearchMode.LEXICAL, None),
            "vector only": (SearchMode.VECTOR, None),
            "hybrid (RRF)": (SearchMode.HYBRID, None),
            "hybrid + term-coverage rerank": (SearchMode.HYBRID, TermCoverageReranker()),
        }
        reports = []
        for name, (mode, reranker) in variants.items():
            retriever = HybridRetriever(
                index,
                embedder,
                reranker=reranker,
                config=HybridConfig(mode=mode),
            )
            reports.append(await evaluate(name, retriever, dataset, chunk_to_key, k=k))
        return reports
    finally:
        if live:
            async with httpx.AsyncClient(trust_env=True) as c:
                for name in cleanup:
                    await c.delete(f"{settings.opensearch_url}/{name}")
            if os_index is not None:
                await os_index.aclose()
        close = getattr(embedder, "aclose", None)
        if close is not None:
            await close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    reports = asyncio.run(run(args.dataset, live=args.live, k=args.k))
    print(format_reports(reports))
    for r in reports:
        for kind, m in r.by_kind.items():
            print(
                f"  {r.name:32s} {kind:12s} n={m.queries:2d} recall={m.recall:.2f} mrr={m.mrr:.2f}"
            )
        for line in r.leaks + r.failures:
            print(f"  !! {r.name}: {line}")
    return 1 if any(r.leaks for r in reports) else 0


if __name__ == "__main__":
    sys.exit(main())
