"""Evaluation release gates: thresholds as data, a pure comparison, a readable table.

A gate fails when the measured value breaks the threshold OR the metric is missing - a gate that
cannot be evaluated must never read as a pass (fail closed). `ops/eval-gates.yaml` holds the
numbers; scripts/eval_gate.py gathers the measurements and calls `evaluate_gates`.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

Op = Literal["min", "max", "eq"]
_OPS = ("min", "max", "eq")


class GateConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Gate:
    metric: str
    op: Op
    value: float
    suite: str
    description: str = ""


@dataclass(frozen=True)
class GateResult:
    gate: Gate
    measured: float | None
    passed: bool
    detail: str


@dataclass(frozen=True)
class GateReport:
    results: tuple[GateResult, ...]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failures(self) -> tuple[GateResult, ...]:
        return tuple(r for r in self.results if not r.passed)


def load_gates(data: Mapping[str, Any]) -> tuple[Gate, ...]:
    """Validate the parsed YAML document. Unknown keys / ops / non-numeric values are errors."""
    if data.get("version") != 1:
        raise GateConfigError("eval-gates: unsupported or missing version")
    raw = data.get("gates")
    if not isinstance(raw, list) or not raw:
        raise GateConfigError("eval-gates: 'gates' must be a non-empty list")
    allowed = {"metric", "op", "value", "suite", "description"}
    gates: list[Gate] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise GateConfigError("eval-gates: each gate must be a mapping")
        extra = set(item) - allowed
        if extra:
            raise GateConfigError(f"eval-gates: unknown keys {sorted(extra)}")
        metric, op, value, suite = (item.get(k) for k in ("metric", "op", "value", "suite"))
        if not isinstance(metric, str) or not metric:
            raise GateConfigError("eval-gates: gate needs a metric name")
        if metric in seen:
            raise GateConfigError(f"eval-gates: duplicate gate for {metric}")
        seen.add(metric)
        if op not in _OPS:
            raise GateConfigError(f"eval-gates: {metric}: op must be one of {_OPS}")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise GateConfigError(f"eval-gates: {metric}: value must be a finite number")
        if not isinstance(suite, str) or not suite:
            raise GateConfigError(f"eval-gates: {metric}: suite is required")
        gates.append(Gate(metric, op, float(value), suite, str(item.get("description", ""))))
    return tuple(gates)


def _check(gate: Gate, measured: float) -> bool:
    if gate.op == "min":
        return measured >= gate.value
    if gate.op == "max":
        return measured <= gate.value
    return measured == gate.value


def evaluate_gates(
    gates: Sequence[Gate], measurements: Mapping[str, float], *, suites: Sequence[str] | None = None
) -> GateReport:
    """Evaluate gates of the given suites (all if None) against flat `suite.metric` measurements."""
    results: list[GateResult] = []
    for gate in gates:
        if suites is not None and gate.suite not in suites:
            continue
        measured = measurements.get(gate.metric)
        if measured is None or not math.isfinite(measured):
            results.append(GateResult(gate, None, False, "missing measurement"))
            continue
        ok = _check(gate, measured)
        results.append(GateResult(gate, measured, ok, "ok" if ok else "threshold broken"))
    return GateReport(tuple(results))


_SYMBOL = {"min": ">=", "max": "<=", "eq": "=="}


def format_report(report: GateReport) -> str:
    rows = [("gate", "required", "measured", "result")]
    for r in report.results:
        required = f"{_SYMBOL[r.gate.op]} {r.gate.value:g}"
        measured = "-" if r.measured is None else f"{r.measured:.4g}"
        rows.append(
            (r.gate.metric, required, measured, "PASS" if r.passed else f"FAIL ({r.detail})")
        )
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines.append("")
    lines.append(
        "RESULT: " + ("PASS" if report.passed else f"FAIL ({len(report.failures)} gate(s))")
    )
    return "\n".join(lines)
