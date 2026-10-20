"""Run the evaluation suites and compare them with ops/eval-gates.yaml. Exit 1 on any failed gate.

    uv run --no-sync python scripts/eval_gate.py                # retrieval (+ agent if present)
    uv run --no-sync python scripts/eval_gate.py --suites retrieval
    uv run --no-sync python scripts/eval_gate.py --runtime-results load.json  # adds runtime gates
    uv run --no-sync python scripts/eval_gate.py --agent-results agent.json  # skip the agent run

Measurements are a flat {"suite.metric": number} map. The agent evaluation (scripts/eval_agent.py
--json) and the load test (scripts/load_test.py --json-out) emit that shape. A gate whose metric is
missing FAILS: a gate that could not be evaluated must never look like a pass.
"""

import argparse
import asyncio
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml
from ekip_observability.evalgate import evaluate_gates, format_report, load_gates
from ekip_retrieval.evaluation import EvalReport

ROOT = Path(__file__).resolve().parent.parent
GATES_PATH = ROOT / "ops" / "eval-gates.yaml"
AGENT_SCRIPT = ROOT / "scripts" / "eval_agent.py"


def slug(name: str) -> str:
    """'hybrid + term-coverage rerank' -> 'hybrid_rerank' (stable metric keys)."""
    known = {
        "bm25 only": "bm25_only",
        "vector only": "vector_only",
        "hybrid (RRF)": "hybrid_rrf",
        "hybrid + term-coverage rerank": "hybrid_rerank",
    }
    return known.get(name) or re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def retrieval_measurements(reports: list[EvalReport]) -> dict[str, float]:
    out: dict[str, float] = {
        "retrieval.leaks": float(sum(len(r.leaks) for r in reports)),
        "retrieval.failures": float(sum(len(r.failures) for r in reports)),
    }
    for r in reports:
        key = f"retrieval.{slug(r.name)}"
        out[f"{key}.recall"] = r.overall.recall
        out[f"{key}.mrr"] = r.overall.mrr
        out[f"{key}.ndcg"] = r.overall.ndcg
        out[f"{key}.hit_rate"] = r.overall.hit_rate
        out[f"{key}.ms_per_query"] = r.mean_latency_ms
    return out


def _load_eval_retrieval() -> Any:
    spec = importlib.util.spec_from_file_location(
        "eval_retrieval", ROOT / "scripts/eval_retrieval.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load scripts/eval_retrieval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_retrieval"] = module
    spec.loader.exec_module(module)
    return module


def run_retrieval(k: int) -> dict[str, float]:
    module = _load_eval_retrieval()
    reports = asyncio.run(module.run(ROOT / module.DEFAULT_DATASET, live=False, k=k))
    return retrieval_measurements(reports)


def parse_flat_json(text: str) -> dict[str, float]:
    """Last JSON object in `text`, keeping only numeric values."""
    start = text.rfind("\n{")
    candidate = text[start + 1 :] if start != -1 else text
    data = json.loads(candidate)
    return {
        k: float(v)
        for k, v in data.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }


def run_agent_eval() -> dict[str, float]:
    proc = subprocess.run(  # noqa: S603 - fixed local script, no user input
        [sys.executable, str(AGENT_SCRIPT), "--json"],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    if proc.returncode != 0:
        print(proc.stderr[-2000:], file=sys.stderr)
        raise SystemExit(f"agent evaluation failed (exit {proc.returncode})")
    return parse_flat_json(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--gates", type=Path, default=GATES_PATH)
    parser.add_argument("--suites", help="comma list of: retrieval,agent,runtime")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--agent-results", type=Path)
    parser.add_argument("--runtime-results", type=Path)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    if args.suites:
        suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    else:
        suites = ["retrieval"]
        if args.agent_results or AGENT_SCRIPT.exists():
            suites.append("agent")
        if args.runtime_results:
            suites.append("runtime")

    measurements: dict[str, float] = {}
    if "retrieval" in suites:
        measurements.update(run_retrieval(args.k))
    if "agent" in suites:
        if args.agent_results:
            measurements.update(parse_flat_json(args.agent_results.read_text()))
        elif AGENT_SCRIPT.exists():
            measurements.update(run_agent_eval())
        # no results and no script: agent gates will report "missing measurement" and FAIL
    if "runtime" in suites and args.runtime_results:
        measurements.update(parse_flat_json(args.runtime_results.read_text()))

    gates = load_gates(yaml.safe_load(args.gates.read_text()))
    report = evaluate_gates(gates, measurements, suites=suites)
    print(format_report(report))
    if args.json_out:
        args.json_out.write_text(json.dumps(measurements, indent=2, sort_keys=True))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
