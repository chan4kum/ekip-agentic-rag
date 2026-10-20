import pytest
from ekip_contracts.tools import ToolRisk
from ekip_tools import RegistryError, ToolRegistry, load_registry_file, registry_from_mapping
from ekip_tools.views import views_from_mapping
from pydantic import ValidationError
from tests_paths import ROOT
from tools_testkit import make_spec, write_spec

REGISTRY_FILE = ROOT / "tools" / "registry.yaml"


def test_the_shipped_registry_loads_and_is_consistent() -> None:
    registry = load_registry_file(REGISTRY_FILE)
    names = {s.name for s in registry.enabled()}
    assert names == {"search_documents", "query_database", "web_search", "create_ticket"}
    assert registry.get("create_ticket") is not None
    assert registry.get("create_ticket").risk is ToolRisk.WRITE  # type: ignore[union-attr]
    assert registry.backends_in_use() == {"mcp_docs", "db_views", "web_search", "ticketing"}


def test_shipped_views_match_the_query_database_enum() -> None:
    import yaml

    data = yaml.safe_load(REGISTRY_FILE.read_text(encoding="utf-8"))
    views = views_from_mapping(data)
    registry = registry_from_mapping(data)
    spec = registry.get("query_database")
    assert spec is not None
    assert set(spec.input_schema["properties"]["view"]["enum"]) == set(views)


def test_write_tools_are_never_registered_for_retry() -> None:
    with pytest.raises(ValidationError):
        write_spec(idempotent=True, max_retries=1)
    with pytest.raises(ValidationError):
        make_spec(max_retries=1, idempotent=False)


def test_duplicate_names_and_bad_schemas_fail_at_load() -> None:
    with pytest.raises(RegistryError):
        ToolRegistry([make_spec("dup_tool"), make_spec("dup_tool")])
    bad = make_spec(
        "bad_tool", input_schema={"type": "object", "properties": {"a": {"type": "string"}}}
    )
    with pytest.raises(RegistryError):
        ToolRegistry([bad])


def test_disabled_tools_do_not_exist() -> None:
    registry = ToolRegistry([make_spec("off_tool", enabled=False), make_spec("on_tool")])
    assert registry.get("off_tool") is None
    assert [s.name for s in registry.enabled()] == ["on_tool"]


@pytest.mark.parametrize("name", ["Bad", "a", "has space", "x" * 70, "1abc"])
def test_tool_names_are_restricted(name: str) -> None:
    with pytest.raises(ValidationError):
        make_spec(name)


def test_a_tool_needs_at_least_one_role() -> None:
    with pytest.raises(ValidationError):
        make_spec(required_roles=())


def test_registry_mapping_needs_tools() -> None:
    with pytest.raises(RegistryError):
        registry_from_mapping({})
