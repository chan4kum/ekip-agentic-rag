"""Compare the Phase 6 single-search agent with the multi-step agent.

    uv run python scripts/eval_agent.py            # offline: scripted planner + citing fake model
    uv run python scripts/eval_agent.py --live     # real planner/answer/judge model from .env
    uv run python scripts/eval_agent.py --require-win

Both variants use the same documents, the same in-memory index and the same caller identities.
Exit code 1 if ANY variant retrieved or cited a passage the caller may not read, or a run crashed.
Exit code 2 (only with --require-win) if multi-step does not beat single-search on cited recall.
Offline mode measures evidence coverage and leaks with a deterministic model; it says nothing
about answer wording. --live adds the real model but still uses the small in-memory dataset.
"""

import argparse
import asyncio
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from ekip_agents import (
    Budget,
    Grounded,
    GroundednessResult,
    LLMGroundednessChecker,
    LLMPlanner,
    RuleBasedRouter,
    build_agent_graph,
)
from ekip_agents.evaluation import (
    AgentEvalReport,
    CountingRetriever,
    ScriptedPlanner,
    evaluate_agent,
    format_agent_reports,
    load_agent_dataset,
)
from ekip_config.settings import load_settings
from ekip_contracts.context import ContextBundle
from ekip_contracts.llm import ChatMessage, ChatModel
from ekip_llm_gateway import HashingEmbedder, OpenAIChatModel
from ekip_retrieval import HybridConfig, HybridRetriever, SearchMode, ingest_dataset
from ekip_search.memory import InMemoryChunkIndex

DEFAULT_DATASET = Path("evaluations/datasets/agent_v1.json")
_SOURCE_ID = re.compile(r'<source id="(\d+)"')


class CitingModel:
    """Deterministic offline model: answers by citing every source it was given."""

    @property
    def model_name(self) -> str:
        return "offline-citing-model"

    async def complete(
        self, messages: Sequence[ChatMessage], *, max_tokens: int = 800, temperature: float = 0.0
    ) -> str:
        numbers = sorted({int(n) for n in _SOURCE_ID.findall(messages[-1].content)})
        if not numbers:
            return "The sources do not answer this."
        return "According to the sources: " + "".join(f"[{n}]" for n in numbers)


class TrustingChecker:
    """Offline stand-in for the judge: keeps the groundedness path exercised."""

    async def check(self, answer: str, context: ContextBundle) -> GroundednessResult:
        return GroundednessResult(Grounded.GROUNDED)


async def run(dataset_path: Path, *, live: bool) -> list[AgentEvalReport]:
    dataset, cases = load_agent_dataset(dataset_path)
    embedder = HashingEmbedder(64)
    index = InMemoryChunkIndex(model_version=embedder.model_version, dimensions=embedder.dimensions)
    await index.ensure_ready()
    chunk_to_key = await ingest_dataset(index, embedder, dataset)
    model: ChatModel = CitingModel()
    opened: OpenAIChatModel | None = None
    if live:
        s = load_settings()
        if s.llm_api_key is None or not s.llm_model:
            raise SystemExit("--live needs LLM_MODEL and LLM_API_KEY in .env")
        opened = OpenAIChatModel(
            s.llm_api_key.get_secret_value(),
            base_url=str(s.llm_base_url),
            model=s.llm_model,
            limit_field=s.llm_limit_field,
            send_temperature=s.llm_send_temperature,
        )
        model = opened
    try:
        reports = []
        for name, multistep in (("single search (phase 6)", False), ("multi-step", True)):
            counter = CountingRetriever(
                HybridRetriever(index, embedder, config=HybridConfig(mode=SearchMode.HYBRID))
            )
            graph = build_agent_graph(
                router=RuleBasedRouter(),
                retriever=counter,
                model=model,
                planner=(LLMPlanner(model) if live else ScriptedPlanner(cases))
                if multistep
                else None,
                grounding=(LLMGroundednessChecker(model) if live else TrustingChecker())
                if multistep
                else None,
                budget=Budget(),
            )
            reports.append(await evaluate_agent(name, graph, counter, dataset, cases, chunk_to_key))
        return reports
    finally:
        await embedder.aclose() if hasattr(embedder, "aclose") else None
        if opened is not None:
            await opened.aclose()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--require-win", action="store_true")
    args = parser.parse_args()
    reports = asyncio.run(run(args.dataset, live=args.live))
    print(format_agent_reports(reports))
    for r in reports:
        for kind, value in r.recall_by_kind.items():
            print(f"  {r.name:24s} {kind:12s} cited recall={value:.2f}")
        for line in (*r.leaks, *r.errors):
            print(f"  !! {r.name}: {line}")
        for line in r.unexpected:
            print(f"  .. {r.name}: {line}")
    if any(r.leaks or r.errors for r in reports):
        return 1
    single, multi = reports
    print(f"\nmulti-step vs single-search cited recall: {multi.recall - single.recall:+.3f}")
    if args.require_win and multi.recall <= single.recall:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
