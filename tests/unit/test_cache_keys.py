import pytest
from cache_testkit import scope
from ekip_caching import CacheVersions, build_key, normalise_query, scope_fingerprint


def test_normalise_collapses_case_space_and_unicode_forms() -> None:
    assert normalise_query("  What IS  the\n torque? ") == "what is the torque?"
    assert normalise_query("ﬁle") == normalise_query("file")  # NFKC ligature


def test_key_is_deterministic_and_never_contains_the_query() -> None:
    parts = [("query", "secret salary question"), ("v", "1")]
    a = build_key(layer="response", tenant_id="acme", parts=parts)
    assert a == build_key(layer="response", tenant_id="acme", parts=parts)
    assert "secret" not in a and "salary" not in a
    assert a.startswith("ekip:cache:response:acme:")


def test_fields_cannot_collide_by_moving_text_between_them() -> None:
    a = build_key(layer="response", tenant_id="acme", parts=[("a", "x"), ("b", "yz")])
    b = build_key(layer="response", tenant_id="acme", parts=[("a", "xy"), ("b", "z")])
    c = build_key(layer="response", tenant_id="acme", parts=[("ax", ""), ("b", "yz")])
    assert len({a, b, c}) == 3


def test_tenant_and_layer_change_the_key() -> None:
    p = [("q", "x")]
    keys = {
        build_key(layer="response", tenant_id="acme", parts=p),
        build_key(layer="response", tenant_id="globex", parts=p),
        build_key(layer="retrieval", tenant_id="acme", parts=p),
    }
    assert len(keys) == 3


@pytest.mark.parametrize("layer", ["", "Resp", "a:b", "x" * 40])
def test_bad_layer_is_rejected(layer: str) -> None:
    with pytest.raises(ValueError):
        build_key(layer=layer, tenant_id="acme", parts=[])


def test_bad_tenant_is_rejected() -> None:
    with pytest.raises(ValueError):
        build_key(layer="response", tenant_id="a:b", parts=[])


def test_scope_fingerprint_binds_user_and_groups_but_not_group_order() -> None:
    assert scope_fingerprint(scope("acme", "alice", "x", "y")) == scope_fingerprint(
        scope("acme", "alice", "y", "x")
    )
    others = {
        scope_fingerprint(scope("acme", "alice", "x")),
        scope_fingerprint(scope("acme", "bob", "x")),
        scope_fingerprint(scope("acme", "alice", "x", "y")),
        scope_fingerprint(scope("acme", "alice")),
        scope_fingerprint(scope("acme", "", "alice")),  # group named like a user
    }
    assert len(others) == 5


def test_versions_must_be_non_empty() -> None:
    with pytest.raises(ValueError, match="model_version"):
        CacheVersions(retrieval_config="r", prompt="p", model=" ", output_policy="o")
