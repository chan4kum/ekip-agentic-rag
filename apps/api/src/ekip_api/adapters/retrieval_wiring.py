"""Choose the retrieval backend from settings (used only by the composition root)."""

from collections.abc import Awaitable, Callable

from ekip_config.settings import Settings
from ekip_contracts.embedding import Embedder
from ekip_contracts.retrieval import Retriever
from ekip_llm_gateway import GeminiEmbedder, HashingEmbedder
from ekip_retrieval import HybridRetriever
from ekip_search.opensearch import OpenSearchChunkIndex

from ekip_api.adapters.demo_data import SeededDemoSearcher

Closer = Callable[[], Awaitable[None]]


def _embedder(settings: Settings) -> Embedder:
    if settings.embedding_provider == "hashing":
        return HashingEmbedder(settings.embedding_dimensions)
    key = settings.google_api_key or settings.gemini_api_key
    if key is None:
        raise ValueError("GOOGLE_API_KEY (or GEMINI_API_KEY) is required for Gemini embeddings")
    return GeminiEmbedder(
        key.get_secret_value(),
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
    )


def build_retriever(settings: Settings) -> tuple[Retriever, list[Closer]]:
    """Returns the retriever and the async closers for any clients it opened."""
    if settings.retrieval_backend == "memory":
        demo = SeededDemoSearcher()
        return HybridRetriever(demo, demo.embedder), []
    embedder = _embedder(settings)
    index = OpenSearchChunkIndex(
        str(settings.opensearch_url),
        model_version=embedder.model_version,
        dimensions=embedder.dimensions,
    )
    closers: list[Closer] = [index.aclose]
    close_embedder = getattr(embedder, "aclose", None)
    if close_embedder is not None:
        closers.append(close_embedder)
    return HybridRetriever(index, embedder), closers
