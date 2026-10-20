"""The static agent registry: the shipped file is valid, and bad files fail loudly."""

import copy
from pathlib import Path
from typing import Any

import pytest
from ekip_agents import AgentRegistry, Budget, RegistryError, load_registry, parse_registry

ROOT = Path(__file__).resolve().parents[2]

GOOD: dict[str, Any] = {
    "version": 1,
    "agents": [
        {
            "id": "retrieval",
            "version": "1.2.3",
            "domain": "knowledge",
            "description": "Searches documents.",
            "capabilities": ["documents.search"],
            "required_permissions": ["documents:read"],
            "approved_tools": ["search_documents"],
            "budget": {"max_retrievals": 3, "max_seconds": 20},
        }
    ],
}


def changed(**agent_fields: Any) -> dict[str, Any]:
    data = copy.deepcopy(GOOD)
    data["agents"][0].update(agent_fields)
    return data


def test_the_shipped_registry_is_valid() -> None:
    registry = load_registry(ROOT / "agents" / "registry.yaml")
    assert len(registry) >= 5
    assert {"router", "planner", "retrieval", "analysis", "verifier"} <= {a.id for a in registry}
    assert registry.allows_tool("retrieval", "search_documents")


def test_the_shipped_registry_never_lets_a_non_action_agent_write() -> None:
    registry = load_registry(ROOT / "agents" / "registry.yaml")
    for agent in registry:
        assert not any(
            t.startswith(("create_", "update_", "delete_")) for t in agent.approved_tools
        )
        if agent.domain == "enterprise-action":
            assert agent.requires_approval


def test_a_good_entry_parses_with_its_budget() -> None:
    registry = parse_registry(GOOD)
    spec = registry.get("retrieval")
    assert spec is not None and spec.version == "1.2.3"
    assert spec.budget == Budget(max_retrievals=3, max_seconds=20)
    assert spec.requires_approval is False
    assert registry.by_domain("knowledge") == (spec,)
    assert registry.with_capability("documents.search") == (spec,)


def test_lookups_fail_closed() -> None:
    registry: AgentRegistry = parse_registry(GOOD)
    assert registry.get("ghost") is None
    assert not registry.allows_tool("ghost", "search_documents")
    assert not registry.allows_tool("retrieval", "delete_everything")


@pytest.mark.parametrize(
    "bad",
    [
        None,
        [],
        {"version": 2, "agents": GOOD["agents"]},
        {"version": 1},
        {"version": 1, "agents": []},
        {"version": 1, "agents": GOOD["agents"], "extra": 1},
        changed(id="Bad Id"),
        changed(id=7),
        changed(version="1.0"),
        changed(domain=""),
        changed(description=""),
        changed(description="x" * 201),
        changed(capabilities=[]),
        changed(capabilities="documents.search"),
        changed(capabilities=["a b"]),
        changed(capabilities=["documents.search", "documents.search"]),
        changed(approved_tools=["Search"]),
        changed(required_permissions=[1]),
        changed(requires_approval="yes"),
        changed(unknown_field=True),
        changed(budget={"max_steps": -1}),
        changed(budget={"max_seconds": 0}),
        changed(budget={"max_bananas": 1}),
        changed(budget={"max_steps": "12"}),
        changed(budget={"max_steps": True}),
        changed(budget={"max_steps": 1.5}),
        changed(budget=[]),
    ],
)
def test_bad_registries_are_rejected(bad: Any) -> None:
    with pytest.raises(RegistryError):
        parse_registry(bad)


def test_duplicate_ids_are_rejected() -> None:
    data = copy.deepcopy(GOOD)
    data["agents"].append(copy.deepcopy(data["agents"][0]))
    with pytest.raises(RegistryError, match="duplicate"):
        parse_registry(data)


def test_a_non_mapping_agent_is_rejected() -> None:
    with pytest.raises(RegistryError):
        parse_registry({"version": 1, "agents": ["retrieval"]})


def test_load_errors_do_not_leak_file_text(tmp_path: Path) -> None:
    broken = tmp_path / "r.yaml"
    broken.write_text("version: 1\nagents: [unterminated SECRET-TEXT", encoding="utf-8")
    with pytest.raises(RegistryError) as info:
        load_registry(broken)
    assert "SECRET-TEXT" not in str(info.value)
    with pytest.raises(RegistryError):
        load_registry(tmp_path / "missing.yaml")


def test_yaml_cannot_construct_python_objects(tmp_path: Path) -> None:
    evil = tmp_path / "r.yaml"
    evil.write_text("version: !!python/object/apply:os.getcwd []\nagents: []\n", encoding="utf-8")
    with pytest.raises(RegistryError):
        load_registry(evil)
