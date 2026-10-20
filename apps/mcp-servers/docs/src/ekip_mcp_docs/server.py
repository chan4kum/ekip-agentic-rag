"""Internal MCP server exposing the enterprise document search as the `search_documents` tool.

This server is a trust boundary of its own. It does NOT trust the gateway's word for who is
calling: it verifies the delegation token in `_meta` (signature, expiry, audience = this tool),
rebuilds the caller's SecurityContext from it, asks the Authorizer, and searches with a
SearchScope derived from that context. The tenant is never read from tool arguments.
"""

import json
from typing import Any

from ekip_contracts.authorization import Action, Authorizer, Resource
from ekip_contracts.identity import SecurityContext
from ekip_contracts.retrieval import Retriever, SearchScope
from ekip_tools.delegation import TokenError, TokenSigner, verify_delegation
from mcp.server.mcpserver import Context, MCPServer

TOOL_NAME = "search_documents"
DELEGATION_META_KEY = "ekip.dev/delegation"
MAX_SNIPPET_CHARS = 1200
MAX_LIMIT = 10
MAX_QUERY_CHARS = 500


class McpAuthError(Exception):
    """Raised inside a tool; MCP turns it into an isError result without internals."""


def _caller(ctx: Context[Any], signer: TokenSigner) -> SecurityContext:
    meta = ctx.request_context.meta or {}
    token = meta.get(DELEGATION_META_KEY)
    if not isinstance(token, str):
        raise McpAuthError("missing delegation")
    try:
        return verify_delegation(signer, token, TOOL_NAME)
    except TokenError as exc:
        raise McpAuthError("invalid delegation") from exc


def build_docs_server(
    *, retriever: Retriever, authorizer: Authorizer, signer: TokenSigner
) -> MCPServer:
    server = MCPServer(
        name="ekip-docs",
        instructions="Search the caller's authorized enterprise documents.",
    )

    @server.tool(name=TOOL_NAME, description="Search authorized enterprise documents.")
    async def search_documents(query: str, ctx: Context[Any], limit: int = 5) -> str:
        caller = _caller(ctx, signer)
        text = query.strip()
        if not text or len(text) > MAX_QUERY_CHARS:
            raise McpAuthError("invalid query")
        decision = await authorizer.authorize(
            caller,
            Action.DOCUMENTS_SEARCH,
            Resource(kind="documents", tenant_id=caller.tenant_id),
        )
        if not decision.allowed:
            raise McpAuthError("not permitted")
        hits = await retriever.search(
            text, scope=SearchScope.from_context(caller), limit=max(1, min(limit, MAX_LIMIT))
        )
        return json.dumps(
            [
                {
                    "chunk_id": h.chunk_id,
                    "document_id": h.document_id,
                    "title": h.title,
                    "source_uri": h.source_uri,
                    "text": h.text[:MAX_SNIPPET_CHARS],
                }
                for h in hits
            ]
        )

    return server
