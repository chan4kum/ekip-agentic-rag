"""Security-critical query fragments, shared by every search the platform will ever run."""

from typing import Any


def tenant_and_acl_filter(
    tenant_id: str, user: str, groups: frozenset[str] | set[str]
) -> list[dict[str, Any]]:
    """The mandatory filter: same tenant AND (listed user OR member of a listed group).

    Fail-closed by construction: a chunk whose ACL lists nobody has empty `allowed_*` fields and so
    matches neither clause. There is no 'public' marker to forget to check.
    """
    should: list[dict[str, Any]] = [{"term": {"allowed_users": user}}]
    if groups:
        should.append({"terms": {"allowed_groups": sorted(groups)}})
    return [
        {"term": {"tenant_id": tenant_id}},
        {"bool": {"should": should, "minimum_should_match": 1}},
    ]
