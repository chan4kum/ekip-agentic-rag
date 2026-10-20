"""query_database: parameterised reads over an ALLOWLIST of named, tenant-scoped views.

There is no free-form SQL anywhere in this path. A view is reviewed config that fixes the
relation, the returned columns and the one column the caller may filter on. The model supplies
only the view NAME (from an enum), a filter value and a row limit; all three travel as bound
parameters. The tenant comes from the SecurityContext and is always applied.
"""

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ekip_contracts.identity import SecurityContext
from ekip_contracts.tools import ToolSpec

from ekip_tools.resilience import ToolBackendError

_IDENT = re.compile(r"[a-z][a-z0-9_]{0,62}")
MAX_ROWS = 50


@dataclass(frozen=True)
class ViewDefinition:
    name: str
    relation: str  # table/view name, e.g. "reporting.open_orders"
    columns: tuple[str, ...]  # the ONLY columns ever returned
    filter_column: str  # the one column `key` is compared with
    tenant_column: str = "tenant_id"
    max_rows: int = MAX_ROWS

    def __post_init__(self) -> None:
        parts = self.relation.split(".")
        names = [self.name, *parts, *self.columns, self.filter_column, self.tenant_column]
        if not (1 <= len(parts) <= 2) or not all(_IDENT.fullmatch(n) for n in names):
            raise ValueError("view definition contains an invalid identifier")
        if self.filter_column not in self.columns or not self.columns:
            raise ValueError("filter_column must be one of the view's columns")
        if not 1 <= self.max_rows <= MAX_ROWS:
            raise ValueError("max_rows out of range")

    def statement(self) -> str:
        """Parameterised SQL: $1 tenant, $2 filter value, $3 row limit. Identifiers were
        validated against a strict pattern above; no caller-supplied text is interpolated."""
        cols = ", ".join(self.columns)
        # identifiers are validated in __post_init__; every value is a bound parameter
        return (
            f"SELECT {cols} FROM {self.relation} "  # noqa: S608
            f"WHERE {self.tenant_column} = $1 AND {self.filter_column} = $2 "
            f"ORDER BY 1 LIMIT $3"
        )


def views_from_mapping(data: Mapping[str, Any]) -> dict[str, ViewDefinition]:
    views: dict[str, ViewDefinition] = {}
    for entry in data.get("views", []):
        view = ViewDefinition(
            name=entry["name"],
            relation=entry["relation"],
            columns=tuple(entry["columns"]),
            filter_column=entry["filter_column"],
            tenant_column=entry.get("tenant_column", "tenant_id"),
            max_rows=int(entry.get("max_rows", MAX_ROWS)),
        )
        views[view.name] = view
    return views


class ViewExecutor(Protocol):
    async def fetch(
        self, statement: str, *, tenant_id: str, key: str, limit: int
    ) -> Sequence[Mapping[str, Any]]:
        """Run `statement` with $1=tenant_id, $2=key, $3=limit as BOUND parameters (asyncpg
        `conn.fetch(statement, tenant_id, key, limit)`), on a read-only connection/role."""
        ...


class ViewQueryBackend:
    def __init__(self, views: Mapping[str, ViewDefinition], executor: ViewExecutor) -> None:
        self._views = dict(views)
        self._executor = executor

    async def call(
        self, context: SecurityContext, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> str:
        view = self._views.get(str(arguments["view"]))
        if view is None:
            raise ToolBackendError("unknown view")
        limit = min(int(arguments.get("limit", 10)), view.max_rows)
        try:
            rows = await self._executor.fetch(
                view.statement(),
                tenant_id=context.tenant_id,  # from the verified context, never the arguments
                key=str(arguments["key"]),
                limit=limit,
            )
        except Exception as exc:
            raise ToolBackendError("query failed", retryable=spec.idempotent) from exc
        projected = [{c: row.get(c) for c in view.columns} for row in rows[:limit]]
        return json.dumps(projected, default=str)
