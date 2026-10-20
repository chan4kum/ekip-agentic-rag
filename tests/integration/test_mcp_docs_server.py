"""End to end over a real (in-process) MCP connection: gateway -> McpToolBackend -> docs server."""

import json
from typing import Any, cast

import pytest
from ekip_authorization.rbac import RbacAuthorizer
from ekip_contracts.retrieval import RetrievedChunk, SearchScope
from ekip_contracts.tools import ToolStatus
from ekip_mcp_docs import build_docs_server
from ekip_ratelimit.memory import InMemoryRateLimiter
from ekip_tools import (
    InMemoryAuditSink,
    McpToolBackend,
    RoleToolAuthorizer,
    TokenSigner,
    ToolGateway,
    load_registry_file,
    mint_delegation,
)
from mcp import Client
from mcp.types import RequestParamsMeta
from tests_paths import ROOT
from tools_testkit import ALICE, KEY, OTHER_TENANT


class SpyRetriever:
    def __init__(self) -> None:
        self.scopes: list[SearchScope] = []

    async def search(
        self, query: str, *, scope: SearchScope, limit: int = 5
    ) -> list[RetrievedChunk]:
        self.scopes.append(scope)
        return [
            RetrievedChunk(
                chunk_id="c1", document_id="d1", text=f"answer to {query}", score=1.0, title="Doc"
            )
        ]


def build() -> tuple[ToolGateway, SpyRetriever, InMemoryAuditSink, TokenSigner, Any]:
    signer = TokenSigner(KEY)
    retriever = SpyRetriever()
    server = build_docs_server(retriever=retriever, authorizer=RbacAuthorizer(), signer=signer)
    registry = load_registry_file(ROOT / "tools" / "registry.yaml")
    audit = InMemoryAuditSink()
    only_docs = type(registry)([s for s in registry.enabled() if s.name == "search_documents"])
    gateway = ToolGateway(
        registry=only_docs,
        backends={"mcp_docs": McpToolBackend(server, signer)},
        authorizer=RoleToolAuthorizer(),
        user_limiter=InMemoryRateLimiter(capacity=50, refill_per_second=5),
        tenant_limiter=InMemoryRateLimiter(capacity=50, refill_per_second=5),
        audit=audit,
    )
    return gateway, retriever, audit, signer, server


async def test_search_documents_through_the_gateway_over_mcp() -> None:
    gateway, retriever, audit, _, _ = build()
    result = await gateway.call(ALICE, "search_documents", {"query": "torque", "limit": 3})
    assert result.status is ToolStatus.OK, result.detail
    assert "answer to torque" in result.content
    # the server derived the scope from the delegated identity, not from any argument
    assert retriever.scopes == [SearchScope(tenant_id="tenant-a", user="alice")]
    assert audit.records[-1].status is ToolStatus.OK


async def test_the_server_rejects_calls_without_a_valid_delegation() -> None:
    _, retriever, _, signer, server = build()
    async with Client(server) as client:
        bare = await client.call_tool("search_documents", {"query": "x"})
        forged = await client.call_tool(
            "search_documents",
            {"query": "x"},
            meta=cast(RequestParamsMeta, {"ekip.dev/delegation": "forged.token"}),
        )
        other_tool = await client.call_tool(
            "search_documents",
            {"query": "x"},
            meta=cast(
                RequestParamsMeta,
                {"ekip.dev/delegation": mint_delegation(signer, ALICE, "query_database")},
            ),
        )
    assert bare.is_error and forged.is_error and other_tool.is_error
    assert retriever.scopes == []


async def test_each_caller_is_scoped_to_their_own_tenant() -> None:
    gateway, retriever, _, _, _ = build()
    assert (await gateway.call(ALICE, "search_documents", {"query": "x"})).ok
    assert (await gateway.call(OTHER_TENANT, "search_documents", {"query": "x"})).ok
    assert [sc.tenant_id for sc in retriever.scopes] == ["tenant-a", "tenant-b"]


async def test_the_server_re_checks_authorization_itself() -> None:
    """A token for a caller whose roles grant no document access is refused by the server's
    own Authorizer even if the gateway were bypassed."""
    _, retriever, _, signer, server = build()
    nobody = ALICE.model_copy(update={"roles": frozenset({"guest"})})
    async with Client(server) as client:
        result = await client.call_tool(
            "search_documents",
            {"query": "x"},
            meta=cast(
                RequestParamsMeta,
                {"ekip.dev/delegation": mint_delegation(signer, nobody, "search_documents")},
            ),
        )
    assert result.is_error
    assert retriever.scopes == []


@pytest.mark.parametrize(
    "arguments", [{"query": ""}, {"query": "x" * 501}, {"query": "q", "tenant_id": "tenant-b"}]
)
async def test_bad_arguments_are_stopped_by_the_gateway(arguments: dict[str, Any]) -> None:
    gateway, retriever, _, _, _ = build()
    result = await gateway.call(ALICE, "search_documents", arguments)
    assert result.status is ToolStatus.INVALID_ARGUMENTS
    assert retriever.scopes == []


async def test_server_output_is_valid_json_list() -> None:
    gateway, _, _, _, _ = build()
    result = await gateway.call(ALICE, "search_documents", {"query": "torque"})
    body = result.content.split("\n", 1)[1].rsplit("\n", 1)[0]
    assert json.loads(body)[0]["chunk_id"] == "c1"
