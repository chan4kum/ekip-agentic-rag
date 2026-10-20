"""Outbound URL checks (SSRF defence) for tools that call out to the network.

Layers, because none is enough alone:
  1. The URL must be https, have no credentials, and use an allowlisted exact host and port.
  2. IP literals and names that resolve to private, loopback, link-local, multicast, reserved or
     metadata addresses are refused (so an allowlisted name that is re-pointed inside is caught).
  3. Redirects are never followed by the callers (a redirect is a new, unchecked URL).
The resolve-then-connect gap (DNS rebinding) cannot be closed in application code; pair this with
a Kubernetes egress NetworkPolicy / proxy that only reaches the allowlisted hosts.
"""

import asyncio
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Collection
from urllib.parse import urlsplit

Resolver = Callable[[str], Awaitable[list[str]]]
DEFAULT_PORT = 443


class OutboundBlockedError(Exception):
    """The URL must not be requested. `reason` is a fixed code, safe to log."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def is_public_ip(address: str) -> bool:
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_outbound_url(url: str, allowed_hosts: Collection[str]) -> str:
    """Static checks. Returns the (lower-cased) host or raises OutboundBlockedError."""
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise OutboundBlockedError("scheme_not_https")
    if parts.username is not None or parts.password is not None:
        raise OutboundBlockedError("credentials_in_url")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise OutboundBlockedError("no_host")
    try:
        port = parts.port or DEFAULT_PORT
    except ValueError as exc:
        raise OutboundBlockedError("bad_port") from exc
    if port != DEFAULT_PORT:
        raise OutboundBlockedError("port_not_allowed")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass  # a name, as expected
    else:
        raise OutboundBlockedError("ip_literal")
    if host not in {h.lower() for h in allowed_hosts}:
        raise OutboundBlockedError("host_not_allowlisted")
    return host


async def system_resolver(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, DEFAULT_PORT, type=socket.SOCK_STREAM)
    return [str(info[4][0]) for info in infos]


async def assert_public_host(host: str, resolver: Resolver = system_resolver) -> None:
    """Every address the name resolves to must be public (blocks internal re-pointing)."""
    try:
        addresses = await resolver(host)
    except OSError as exc:
        raise OutboundBlockedError("unresolvable") from exc
    if not addresses or not all(is_public_ip(a) for a in addresses):
        raise OutboundBlockedError("non_public_address")
