"""The agent evaluation: dataset loads, the offline gate holds, a leak fails the run."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
from ekip_agents import RuleBasedRouter, build_agent_graph
from ekip_agents.evaluation import (
    CountingRetriever,
    evaluate_agent,
    load_agent_dataset,
)
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from ekip_llm_gateway import HashingEmbedder
from ekip_retrieval import ingest_dataset
from ekip_search.memory import InMemoryChunkIndex

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "evaluations" / "datasets" / "agent_v1.json"


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("eval_agent", ROOT / "scripts" / "eval_agent.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_agent"] = module
    spec.loader.exec_module(module)
    return module


def test_the_dataset_covers_every_required_kind() -> None:
    dataset, cases = load_agent_dataset(DATASET)
    kinds = {c.query.kind for c in cases}
    assert {"comparison", "multihop", "injection", "restricted", "cross_tenant"} <= kinds
    assert "no_evidence" in kinds
    assert dataset.version == "agent-v1"
    assert any(c.plan and any(t.depends_on for t in c.plan.subtasks) for c in cases)


def test_a_bad_expectation_is_rejected(tmp_path: Path) -> None:
    broken = tmp_path / "d.json"
    broken.write_text(
        DATASET.read_text(encoding="utf-8").replace('"expect": "answered"', '"expect": "maybe"', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expectation"):
        load_agent_dataset(broken)


async def test_the_offline_gate_has_no_leaks_and_multistep_does_not_regress() -> None:
    script = load_script()
    single, multi = await script.run(DATASET, live=False)
    assert single.leaks == () and multi.leaks == ()
    assert single.errors == () and multi.errors == ()
    assert multi.recall >= single.recall
    assert multi.recall_by_kind["comparison"] >= single.recall_by_kind["comparison"]
    assert multi.mean_model_calls > single.mean_model_calls  # the extra cost is visible


def test_the_script_exit_codes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = load_script()
    monkeypatch.setattr(sys, "argv", ["eval_agent.py", "--dataset", str(DATASET)])
    assert script.main() == 0
    assert "multi-step" in capsys.readouterr().out


class LeakyRetriever:
    """Ignores the caller's scope and returns every passage: the leak check must catch it."""

    def __init__(self, everything: list[RetrievedChunk]) -> None:
        self._everything = everything

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        return self._everything[:limit]


async def test_a_retriever_that_ignores_access_control_is_reported_as_leaking() -> None:
    dataset, cases = load_agent_dataset(DATASET)
    embedder = HashingEmbedder(64)
    index = InMemoryChunkIndex(model_version=embedder.model_version, dimensions=embedder.dimensions)
    await index.ensure_ready()
    chunk_to_key = await ingest_dataset(index, embedder, dataset)
    hr = [
        RetrievedChunk(chunk_id=cid, document_id="x", text=f"text {cid}", score=1.0, title="T")
        for cid, key in chunk_to_key.items()
        if key.startswith(("hr-", "b-brake"))
    ]
    counter = CountingRetriever(LeakyRetriever(hr))
    script = load_script()
    graph = build_agent_graph(
        router=RuleBasedRouter(), retriever=counter, model=script.CitingModel()
    )
    report = await evaluate_agent("leaky", graph, counter, dataset, cases, chunk_to_key)
    assert report.leaks  # restricted and cross-tenant passages were returned to alice
    assert any(line.startswith("r1 ->") for line in report.leaks)
