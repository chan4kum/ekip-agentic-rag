"""Reciprocal rank fusion and query hygiene: small pure functions, tested exactly."""

import pytest
from ekip_contracts.retrieval import RetrievedChunk
from ekip_retrieval import InvalidQueryError, normalize_query, reciprocal_rank_fusion


def hit(cid: str, score: float = 1.0, text: str = "t") -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id="d", text=text, score=score)


def ids(hits: list[RetrievedChunk]) -> list[str]:
    return [h.chunk_id for h in hits]


def test_a_chunk_in_both_lists_beats_chunks_in_one() -> None:
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("c"), hit("b")]])
    assert ids(fused)[0] == "b"  # rank 2 + rank 2 beats rank 1 + nothing
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 62)


def test_scores_are_the_standard_rrf_formula() -> None:
    fused = reciprocal_rank_fusion([[hit("a"), hit("b")], [hit("b"), hit("a")]], k=60)
    scores = {h.chunk_id: h.score for h in fused}
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)


def test_raw_scores_do_not_matter_only_ranks() -> None:
    huge = [hit("a", 9999.0), hit("b", 5000.0)]
    tiny = [hit("b", 0.002), hit("a", 0.001)]
    assert ids(reciprocal_rank_fusion([huge, tiny])) == ["a", "b"]  # tie -> best rank, then id


def test_weights_shift_the_balance() -> None:
    lexical, vector = [hit("a"), hit("b")], [hit("b"), hit("a")]
    assert ids(reciprocal_rank_fusion([lexical, vector], weights=[3.0, 1.0]))[0] == "a"
    assert ids(reciprocal_rank_fusion([lexical, vector], weights=[1.0, 3.0]))[0] == "b"


def test_duplicates_within_one_list_count_once() -> None:
    fused = reciprocal_rank_fusion([[hit("a"), hit("a"), hit("b")]])
    assert ids(fused) == ["a", "b"] and fused[0].score == pytest.approx(1 / 61)


def test_data_comes_from_the_first_list_that_has_the_chunk() -> None:
    fused = reciprocal_rank_fusion(
        [[hit("a", text="from lexical")], [hit("a", text="from vector")]]
    )
    assert fused[0].text == "from lexical"


def test_output_order_is_deterministic_on_ties() -> None:
    one = reciprocal_rank_fusion([[hit("b")], [hit("a")]])
    two = reciprocal_rank_fusion([[hit("b")], [hit("a")]])
    assert ids(one) == ids(two) == ["a", "b"]  # equal score and rank -> chunk id


def test_empty_inputs() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    assert ids(reciprocal_rank_fusion([[], [hit("a")]])) == ["a"]


@pytest.mark.parametrize(
    "kwargs", [{"k": 0}, {"weights": [1.0]}, {"weights": [1.0, 0.0]}, {"weights": [1.0, -1.0]}]
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[hit("a")], [hit("b")]], **kwargs)  # type: ignore[arg-type]


# ---- query hygiene ---------------------------------------------------------------------------


def test_queries_are_normalised() -> None:
    assert normalize_query("  hydraulic \n\t pump  ") == "hydraulic pump"
    assert normalize_query("ＡＭＭ-32-11") == "AMM-32-11"  # full-width forms fold to ASCII
    assert normalize_query("pump\x00\x07check") == "pump check"  # control characters become spaces


@pytest.mark.parametrize("raw", ["", "   ", "\x00\x01", "​"])
def test_empty_after_cleaning_is_rejected(raw: str) -> None:
    with pytest.raises(InvalidQueryError):
        normalize_query(raw)


def test_length_is_measured_after_cleaning() -> None:
    assert len(normalize_query("a " * 400)) == 799  # trailing space trimmed, so within the limit
    with pytest.raises(InvalidQueryError):
        normalize_query("x" * 1001)
    assert len(normalize_query("x" * 1000)) == 1000
