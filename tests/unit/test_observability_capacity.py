import pytest
from ekip_observability import CorpusAssumptions, backfill_hours, replicated_vector_gb


def test_masterplan_section_6_numbers() -> None:
    a = CorpusAssumptions()
    assert a.total_chunks == 20_000_000
    assert a.raw_vector_bytes == 81_920_000_000
    assert a.raw_vector_gb == pytest.approx(81.92)
    assert a.max_changed_chunks_per_day == 200_000


def test_dimensions_and_precision_scale_linearly() -> None:
    half = CorpusAssumptions(embedding_dimensions=512)
    assert half.raw_vector_gb == pytest.approx(40.96)
    fp16 = CorpusAssumptions(bytes_per_dimension=2)
    assert fp16.raw_vector_gb == pytest.approx(40.96)


def test_replicas_multiply_raw_vector_bytes_only() -> None:
    assert replicated_vector_gb(CorpusAssumptions(), replicas=1) == pytest.approx(163.84)
    with pytest.raises(ValueError):
        replicated_vector_gb(CorpusAssumptions(), replicas=-1)


def test_backfill_time_needs_a_measured_rate() -> None:
    assert backfill_hours(20_000_000, 1000) == pytest.approx(5.555, rel=1e-3)
    with pytest.raises(ValueError):
        backfill_hours(1, 0)


def test_negative_assumptions_rejected() -> None:
    with pytest.raises(ValueError):
        CorpusAssumptions(documents=-1)
