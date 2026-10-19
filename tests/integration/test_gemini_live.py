"""Live Gemini check. Runs only with RUN_LIVE_LLM=1 and GOOGLE_API_KEY in the environment/.env."""

import math
import os

import pytest
from ekip_config.settings import Settings
from ekip_contracts.embedding import EmbeddingTask
from ekip_llm_gateway import GeminiEmbedder

pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_LLM") != "1", reason="set RUN_LIVE_LLM=1")


async def test_live_embedding_has_configured_dimensions_and_unit_length() -> None:
    s = Settings()
    assert s.google_api_key is not None, "GOOGLE_API_KEY missing"
    e = GeminiEmbedder(
        s.google_api_key.get_secret_value(),
        model=s.embedding_model,
        dimensions=s.embedding_dimensions,
    )
    try:
        docs = await e.embed(
            ["landing gear maintenance procedure", "quarterly sales forecast"],
            task=EmbeddingTask.DOCUMENT,
        )
        [q] = await e.embed(["how do I service the landing gear?"], task=EmbeddingTask.QUERY)
    finally:
        await e.aclose()
    assert all(len(v) == s.embedding_dimensions for v in docs)
    assert math.isclose(math.sqrt(sum(x * x for x in q)), 1.0, rel_tol=1e-6)
    sim = [sum(a * b for a, b in zip(q, d, strict=True)) for d in docs]
    assert sim[0] > sim[1]  # the related document ranks higher
