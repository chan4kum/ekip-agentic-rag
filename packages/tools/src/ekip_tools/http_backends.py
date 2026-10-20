"""Backends for tools that call HTTP APIs: create_ticket (write) and web_search (read).

Both take an injected `httpx.AsyncClient` (tests use `httpx.MockTransport`; production wires a
pooled client with timeouts), never follow redirects, and cap response size. Credentials come
from a `CredentialProvider` (token exchange); the user's own token is never used.
"""

import json
from collections.abc import Callable, Collection, Mapping
from typing import Any, Protocol

import httpx
from ekip_contracts.identity import SecurityContext
from ekip_contracts.tools import ToolSpec

from ekip_tools.resilience import ToolBackendError
from ekip_tools.ssrf import (
    OutboundBlockedError,
    Resolver,
    assert_public_host,
    check_outbound_url,
    system_resolver,
)

MAX_RESPONSE_BYTES = 256 * 1024


class CredentialProvider(Protocol):
    async def token_for(self, context: SecurityContext, audience: str) -> str:
        """A token for `audience` acting for `context` (OAuth 2.0 token exchange, RFC 8693)."""
        ...


async def _read_capped(response: httpx.Response) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk)
        if len(body) > MAX_RESPONSE_BYTES:
            raise ToolBackendError("response too large")
    return bytes(body)


async def _send(
    client: httpx.AsyncClient, request: httpx.Request, *, retryable: bool
) -> tuple[httpx.Response, bytes]:
    try:
        response = await client.send(request, stream=True, follow_redirects=False)
    except httpx.HTTPError as exc:
        raise ToolBackendError("upstream unreachable", retryable=retryable) from exc
    try:
        body = await _read_capped(response)
    finally:
        await response.aclose()
    if response.is_redirect:
        raise ToolBackendError("unexpected redirect")
    if response.status_code >= 500:
        raise ToolBackendError("upstream error", retryable=retryable)
    if response.status_code >= 400:
        raise ToolBackendError(f"upstream rejected the request ({response.status_code})")
    return response, body


def _json(body: bytes) -> Any:
    try:
        return json.loads(body)
    except ValueError as exc:
        raise ToolBackendError("upstream returned invalid JSON") from exc


class TicketBackend:
    """create_ticket: opens an issue in the tenant's ticketing project. WRITE: the gateway has
    already obtained a human approval. The project is derived from the caller's tenant on the
    server side; the model cannot pick it."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str,
        allowed_hosts: Collection[str],
        project_for_tenant: Mapping[str, str],
        credentials: CredentialProvider,
        resolver: Resolver = system_resolver,
    ) -> None:
        self._host = check_outbound_url(base_url, allowed_hosts)
        self._base = base_url.rstrip("/")
        self._client = client
        self._projects = dict(project_for_tenant)
        self._credentials = credentials
        self._resolver = resolver

    async def call(
        self, context: SecurityContext, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> str:
        project = self._projects.get(context.tenant_id)
        if project is None:
            raise ToolBackendError("ticketing is not configured for this tenant")
        try:
            await assert_public_host(self._host, self._resolver)
        except OutboundBlockedError as exc:
            raise ToolBackendError("destination not allowed") from exc
        token = await self._credentials.token_for(context, self._host)
        request = self._client.build_request(
            "POST",
            f"{self._base}/projects/{project}/issues",
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
            json={
                "title": arguments["title"],
                "description": arguments.get("description", ""),
                "priority": arguments.get("priority", "medium"),
                "reporter": context.subject,
            },
        )
        _response, body = await _send(self._client, request, retryable=False)
        data = _json(body)
        key = data.get("key") if isinstance(data, dict) else None
        return f"created ticket {key}" if isinstance(key, str) else "ticket created"


class WebSearchBackend:
    """web_search: queries an allowlisted search API and returns title/url/snippet lines.
    Result URLs are returned as text only; this backend never fetches them.

    `query_filter` is the hook for the PII/secret guardrail: it can rewrite the outgoing query or
    raise ToolBackendError to block it (an internal question must not leak to a public service).
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        endpoint: str,
        allowed_hosts: Collection[str],
        credentials: CredentialProvider,
        query_filter: Callable[[str], str] | None = None,
        resolver: Resolver = system_resolver,
    ) -> None:
        self._host = check_outbound_url(endpoint, allowed_hosts)
        self._endpoint = endpoint
        self._client = client
        self._credentials = credentials
        self._query_filter = query_filter
        self._resolver = resolver

    async def call(
        self, context: SecurityContext, spec: ToolSpec, arguments: Mapping[str, Any]
    ) -> str:
        query = str(arguments["query"])
        if self._query_filter is not None:
            query = self._query_filter(query)
        try:
            await assert_public_host(self._host, self._resolver)
        except OutboundBlockedError as exc:
            raise ToolBackendError("destination not allowed") from exc
        token = await self._credentials.token_for(context, self._host)
        request = self._client.build_request(
            "GET",
            self._endpoint,
            params={"q": query, "count": int(arguments.get("count", 5))},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
        _response, body = await _send(self._client, request, retryable=spec.idempotent)
        data = _json(body)
        results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(results, list):
            raise ToolBackendError("upstream returned an unexpected shape")
        lines = []
        for item in results[:10]:
            if isinstance(item, dict):
                title = str(item.get("title", ""))[:200]
                url = str(item.get("url", ""))[:300]
                snippet = str(item.get("snippet", ""))[:500]
                lines.append(f"- {title} ({url}): {snippet}")
        return "\n".join(lines) if lines else "no results"
