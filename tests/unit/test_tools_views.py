from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from ekip_tools import ToolBackendError, ViewDefinition, ViewQueryBackend, views_from_mapping
from tools_testkit import ALICE, make_spec

VIEW = ViewDefinition(
    name="open_orders",
    relation="reporting.open_orders",
    columns=("order_id", "customer_id", "status"),
    filter_column="customer_id",
)
SPEC = make_spec("query_database", idempotent=True)


class RecordingExecutor:
    def __init__(self, rows: Sequence[Mapping[str, Any]] = ()) -> None:
        self.rows = rows
        self.calls: list[dict[str, Any]] = []

    async def fetch(
        self, statement: str, *, tenant_id: str, key: str, limit: int
    ) -> Sequence[Mapping[str, Any]]:
        self.calls.append(
            {"statement": statement, "tenant_id": tenant_id, "key": key, "limit": limit}
        )
        return self.rows


def test_statement_is_fully_parameterised() -> None:
    assert VIEW.statement() == (
        "SELECT order_id, customer_id, status FROM reporting.open_orders "
        "WHERE tenant_id = $1 AND customer_id = $2 ORDER BY 1 LIMIT $3"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"relation": "orders; DROP TABLE x"},
        {"relation": "a.b.c"},
        {"columns": ("order_id", "1=1 --"), "filter_column": "order_id"},
        {"filter_column": "not_a_column"},
        {"tenant_column": "tenant_id OR 1=1"},
        {"name": "Bad Name"},
        {"max_rows": 500},
    ],
)
def test_view_definitions_reject_injectable_identifiers(kwargs: dict[str, Any]) -> None:
    base: dict[str, Any] = {
        "name": "v",
        "relation": "reporting.v",
        "columns": ("order_id", "customer_id"),
        "filter_column": "customer_id",
    }
    with pytest.raises(ValueError, match="invalid|filter_column|max_rows"):
        ViewDefinition(**{**base, **kwargs})


async def test_tenant_comes_from_the_context_and_columns_are_projected() -> None:
    executor = RecordingExecutor(
        [{"order_id": "o1", "customer_id": "c1", "status": "open", "secret": "x"}]
    )
    backend = ViewQueryBackend({"open_orders": VIEW}, executor)
    out = await backend.call(ALICE, SPEC, {"view": "open_orders", "key": "c1", "limit": 500})
    assert executor.calls[0]["tenant_id"] == "tenant-a"
    assert executor.calls[0]["limit"] == 50  # capped by the view
    assert "secret" not in out
    assert "o1" in out


async def test_unknown_views_and_executor_failures_are_safe() -> None:
    backend = ViewQueryBackend({"open_orders": VIEW}, RecordingExecutor())
    with pytest.raises(ToolBackendError, match="unknown view"):
        await backend.call(ALICE, SPEC, {"view": "users", "key": "c1"})

    class Boom:
        async def fetch(self, *a: Any, **k: Any) -> Sequence[Mapping[str, Any]]:
            raise RuntimeError("relation does not exist: secrets.internal")

    with pytest.raises(ToolBackendError) as err:
        await ViewQueryBackend({"open_orders": VIEW}, Boom()).call(
            ALICE, SPEC, {"view": "open_orders", "key": "c1"}
        )
    assert "secrets" not in err.value.safe_message


def test_views_load_from_mapping() -> None:
    views = views_from_mapping(
        {
            "views": [
                {"name": "v_one", "relation": "r.t", "columns": ["a", "b"], "filter_column": "a"}
            ]
        }
    )
    assert views["v_one"].tenant_column == "tenant_id"
