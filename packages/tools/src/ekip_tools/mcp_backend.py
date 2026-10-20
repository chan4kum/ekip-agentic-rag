"""Backend that runs a tool on an MCP server (in-process for tests, Streamable HTTP in prod).

Per the 2026-07-28 MCP spec the protocol is stateless: there is no session to keep alive, and
every request carries its own `_meta`. We use that channel for two things:
  - `traceparent`: so the tool server's spans join the request's trace;
  - `ekip.dev/delegation`: a short-lived token scoped to THIS tool and caller (see delegation.py).
The user's access token is never forwarded. The server must verify the delegation itself and
re-check authorization (defence in depth); the gateway's checks are not the only line.
"""

from collections.abc import Mapping
from typing import Any, cast

from ekip_contracts.identity import SecurityContext
from ekip_contracts.tools import ToolSpec
from mcp import Client
from mcp.client._transport import Transport
from mcp.server import Server
from mcp.server.mcpserver import MCPServer
from mcp.types import RequestParamsMeta, TextContent
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from ekip_tools.delegation import TokenSigner, mint_delegation
from ekip_tools.resilience import ToolBackendError

DELEGATION_META_KEY = "ekip.dev/delegation"
MCP_CALL_TIMEOUT_S = 30.0

McpTarget = Server[Any] | MCPServer | Transport | str


class McpToolBackend:
    """`target` is an `MCPServer` (tests), a Streamable HTTP URL, or any MCP `Transport`."""

    def __init__(
        self, target: McpTarget, signer: TokenSigner, *, delegation_ttl_s: int = 30
    ) -> None:
        self._target = target
        self._signer = signer
        self._ttl = delegation_ttl_s
        self._propagator = TraceContextTextMapPropagator()

    async def call(
        self, context: SecurityContext, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> str:
        carrier: dict[str, str] = {}
        self._propagator.inject(carrier)
        meta: dict[str, Any] = {
            DELEGATION_META_KEY: mint_delegation(self._signer, context, spec.name, ttl_s=self._ttl),
            **carrier,
        }
        try:
            async with Client(self._target) as client:
                result = await client.call_tool(
                    spec.name,
                    dict(arguments),
                    read_timeout_seconds=min(spec.timeout_s, MCP_CALL_TIMEOUT_S),
                    meta=cast(RequestParamsMeta, meta),
                )
        except TimeoutError:
            raise
        except Exception as exc:
            # Transport/protocol failure: transient, safe to retry only if the tool is idempotent.
            raise ToolBackendError("tool server unavailable", retryable=True) from exc
        text = "\n".join(b.text for b in result.content if isinstance(b, TextContent))
        if result.is_error:
            # The server's own error text is tool output (untrusted); keep the model-facing
            # message generic. The gateway audits the failure.
            raise ToolBackendError("tool reported an error")
        return text
