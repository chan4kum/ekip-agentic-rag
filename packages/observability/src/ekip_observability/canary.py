"""Canary comparison: should a candidate (new prompt / model / reranker) replace the baseline?

Deliberately simple and conservative:
- a minimum sample size per arm, otherwise the answer is HOLD (not enough evidence);
- success-rate metrics use a one-sided NON-INFERIORITY test on the difference of two proportions
  (normal approximation): promote only if we are confident the candidate is not worse than the
  baseline by more than `margin`; roll back if we are confident it IS worse than that margin;
  otherwise hold and keep collecting;
- latency compares the p95 of both arms against a maximum ratio (no distributional claims).
These are release-gate heuristics, not a replacement for judgement on small differences.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ekip_observability.stats import percentile

# One-sided z for alpha = 0.05.
Z_ONE_SIDED_95 = 1.645


class Verdict(StrEnum):
    PROMOTE = "promote"
    HOLD = "hold"
    ROLLBACK = "rollback"


@dataclass(frozen=True)
class RateSample:
    successes: int
    total: int

    def __post_init__(self) -> None:
        if self.total < 0 or not 0 <= self.successes <= self.total:
            raise ValueError("need 0 <= successes <= total")

    @property
    def rate(self) -> float:
        return self.successes / self.total if self.total else 0.0


@dataclass(frozen=True)
class CanaryDecision:
    verdict: Verdict
    reason: str
    delta: float | None = None  # candidate rate - baseline rate
    lower_bound: float | None = None
    upper_bound: float | None = None


def compare_rates(
    baseline: RateSample,
    candidate: RateSample,
    *,
    min_samples: int = 200,
    margin: float = 0.02,
    z: float = Z_ONE_SIDED_95,
) -> CanaryDecision:
    """Higher is better (task success, groundedness pass rate, citation validity...)."""
    if margin < 0:
        raise ValueError("margin must be >= 0")
    if baseline.total < min_samples or candidate.total < min_samples:
        return CanaryDecision(
            Verdict.HOLD,
            f"insufficient data: need >= {min_samples} per arm "
            f"(baseline {baseline.total}, candidate {candidate.total})",
        )
    p0, p1 = baseline.rate, candidate.rate
    delta = p1 - p0
    se = math.sqrt(p0 * (1 - p0) / baseline.total + p1 * (1 - p1) / candidate.total)
    lower = delta - z * se
    upper = delta + z * se
    if upper < -margin:
        return CanaryDecision(
            Verdict.ROLLBACK,
            f"candidate is worse than baseline by more than {margin:.1%} (delta {delta:+.1%})",
            delta,
            lower,
            upper,
        )
    if lower > -margin:
        return CanaryDecision(
            Verdict.PROMOTE,
            f"candidate is not worse than baseline by more than {margin:.1%} (delta {delta:+.1%})",
            delta,
            lower,
            upper,
        )
    return CanaryDecision(
        Verdict.HOLD,
        f"inconclusive (delta {delta:+.1%}, interval [{lower:+.1%}, {upper:+.1%}])",
        delta,
        lower,
        upper,
    )


def compare_latency_p95(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    min_samples: int = 200,
    max_ratio: float = 1.2,
) -> CanaryDecision:
    """Lower is better. Rolls back if the candidate p95 exceeds baseline p95 * max_ratio."""
    if len(baseline) < min_samples or len(candidate) < min_samples:
        return CanaryDecision(
            Verdict.HOLD,
            f"insufficient data: need >= {min_samples} per arm "
            f"(baseline {len(baseline)}, candidate {len(candidate)})",
        )
    base95 = percentile(baseline, 95)
    cand95 = percentile(candidate, 95)
    if base95 <= 0:
        return CanaryDecision(Verdict.HOLD, "baseline p95 is not positive; cannot form a ratio")
    ratio = cand95 / base95
    if ratio > max_ratio:
        return CanaryDecision(
            Verdict.ROLLBACK, f"p95 regressed x{ratio:.2f} (limit x{max_ratio:.2f})", ratio
        )
    return CanaryDecision(Verdict.PROMOTE, f"p95 within limit (x{ratio:.2f})", ratio)


def combine(decisions: Sequence[CanaryDecision]) -> Verdict:
    """Any rollback wins; any hold blocks promotion; promote only if every check promotes."""
    if not decisions:
        return Verdict.HOLD
    verdicts = {d.verdict for d in decisions}
    if Verdict.ROLLBACK in verdicts:
        return Verdict.ROLLBACK
    if Verdict.HOLD in verdicts:
        return Verdict.HOLD
    return Verdict.PROMOTE
