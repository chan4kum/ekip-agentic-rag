import pytest
from ekip_observability import (
    RateSample,
    Verdict,
    combine,
    compare_latency_p95,
    compare_rates,
    percentile,
)


def test_too_few_samples_holds() -> None:
    d = compare_rates(RateSample(90, 100), RateSample(100, 100))
    assert d.verdict is Verdict.HOLD and "insufficient" in d.reason


def test_equal_performance_with_enough_data_promotes() -> None:
    d = compare_rates(RateSample(900, 1000), RateSample(905, 1000), margin=0.02)
    assert d.verdict is Verdict.PROMOTE
    assert d.delta == pytest.approx(0.005)


def test_clear_regression_rolls_back() -> None:
    d = compare_rates(RateSample(900, 1000), RateSample(800, 1000), margin=0.02)
    assert d.verdict is Verdict.ROLLBACK


def test_noisy_small_difference_holds() -> None:
    d = compare_rates(RateSample(180, 200), RateSample(176, 200), margin=0.02)
    assert d.verdict is Verdict.HOLD
    assert d.lower_bound is not None and d.upper_bound is not None


def test_improvement_promotes() -> None:
    assert compare_rates(RateSample(800, 1000), RateSample(900, 1000)).verdict is Verdict.PROMOTE


def test_margin_zero_requires_strict_non_inferiority() -> None:
    d = compare_rates(RateSample(900, 1000), RateSample(899, 1000), margin=0.0)
    assert d.verdict is Verdict.HOLD


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        RateSample(5, 3)
    with pytest.raises(ValueError):
        compare_rates(RateSample(1, 1), RateSample(1, 1), margin=-1)


def test_latency_regression_rolls_back_and_ok_promotes() -> None:
    base = [1.0 + (i % 10) * 0.1 for i in range(300)]
    slower = [x * 1.5 for x in base]
    same = [x * 1.05 for x in base]
    assert compare_latency_p95(base, slower).verdict is Verdict.ROLLBACK
    assert compare_latency_p95(base, same).verdict is Verdict.PROMOTE
    assert compare_latency_p95(base[:10], same).verdict is Verdict.HOLD


def test_combine_rollback_beats_hold_beats_promote() -> None:
    promote = compare_rates(RateSample(900, 1000), RateSample(905, 1000))
    hold = compare_rates(RateSample(9, 10), RateSample(9, 10))
    rollback = compare_rates(RateSample(900, 1000), RateSample(700, 1000))
    assert combine([promote, promote]) is Verdict.PROMOTE
    assert combine([promote, hold]) is Verdict.HOLD
    assert combine([promote, hold, rollback]) is Verdict.ROLLBACK
    assert combine([]) is Verdict.HOLD


def test_percentile() -> None:
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([7], 99) == 7
    assert percentile([1, 2, 3, 4, 5], 100) == 5
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1], 101)
