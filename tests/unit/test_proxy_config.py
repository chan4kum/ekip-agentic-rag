"""Infra guards for the reverse proxy and network layout (no Docker needed)."""

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(rel: str) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / rel).read_text())
    return data


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return load("docker-compose.yml")


@pytest.fixture(scope="module")
def static() -> dict[str, Any]:
    return load("infrastructure/traefik/traefik.yml")


@pytest.fixture(scope="module")
def dynamic() -> dict[str, Any]:
    return load("infrastructure/traefik/dynamic/api.yml")


def test_only_traefik_publishes_an_app_facing_port(compose: dict[str, Any]) -> None:
    assert "ports" not in compose["services"]["api"]
    assert compose["services"]["traefik"]["ports"] == ["127.0.0.1:8080:8080"]


def test_app_network_is_internal_and_api_is_only_on_it(compose: dict[str, Any]) -> None:
    assert compose["networks"]["app"]["internal"] is True
    assert compose["services"]["api"]["networks"] == ["app"]


def test_no_service_mounts_the_docker_socket(compose: dict[str, Any]) -> None:
    for name, svc in compose["services"].items():
        for vol in svc.get("volumes", []):
            assert "docker.sock" not in str(vol), f"{name} mounts the Docker socket"


def test_traefik_uses_file_provider_and_no_dashboard(static: dict[str, Any]) -> None:
    assert "file" in static["providers"]
    assert "docker" not in static["providers"]
    assert static["api"]["dashboard"] is False


def test_untrusted_forwarded_headers_are_not_accepted(static: dict[str, Any]) -> None:
    assert static["entryPoints"]["web"]["forwardedHeaders"]["insecure"] is False


def test_access_log_drops_headers_by_default(static: dict[str, Any]) -> None:
    assert static["accessLog"]["fields"]["headers"]["defaultMode"] == "drop"


def test_api_router_has_required_middlewares(dynamic: dict[str, Any]) -> None:
    mws = dynamic["http"]["routers"]["api"]["middlewares"]
    assert {"secure-headers", "body-limit", "rate-limit"} <= set(mws)
    defined = dynamic["http"]["middlewares"]
    assert all(m in defined for m in mws)


def test_docs_are_not_routed_at_the_edge(dynamic: dict[str, Any]) -> None:
    for router in dynamic["http"]["routers"].values():
        assert "/docs" not in router["rule"]
        assert "/openapi.json" not in router["rule"]


def test_api_trusts_forwarded_ips_only_from_traefik(compose: dict[str, Any]) -> None:
    traefik_ip = compose["services"]["traefik"]["networks"]["app"]["ipv4_address"]
    assert compose["services"]["api"]["environment"]["FORWARDED_ALLOW_IPS"] == traefik_ip


def test_traefik_is_hardened(compose: dict[str, Any]) -> None:
    t = compose["services"]["traefik"]
    assert t["user"] != "0" and not str(t["user"]).startswith("0:")
    assert t["read_only"] is True
    assert t["cap_drop"] == ["ALL"]


def test_every_router_middleware_is_defined(dynamic: dict[str, Any]) -> None:
    defined = dynamic["http"]["middlewares"]
    for router in dynamic["http"]["routers"].values():
        assert all(m in defined for m in router["middlewares"])


def test_retry_is_limited_to_network_errors_and_idempotent_methods(
    dynamic: dict[str, Any],
) -> None:
    retry = dynamic["http"]["middlewares"]["retry"]["retry"]
    assert retry["attempts"] <= 3
    assert "status" not in retry  # retrying 5xx responses is how retry storms start
    assert not retry.get("retryNonIdempotentMethod", False)


def test_circuit_breaker_has_an_expression_and_open_status(dynamic: dict[str, Any]) -> None:
    cb = dynamic["http"]["middlewares"]["circuit-breaker"]["circuitBreaker"]
    assert cb["expression"].strip()
    assert cb["responseCode"] == 503


def test_in_flight_limit_is_positive(dynamic: dict[str, Any]) -> None:
    assert dynamic["http"]["middlewares"]["in-flight"]["inFlightReq"]["amount"] > 0


def test_upstream_timeouts_are_set(dynamic: dict[str, Any]) -> None:
    name = dynamic["http"]["services"]["api"]["loadBalancer"]["serversTransport"]
    timeouts = dynamic["http"]["serversTransports"][name]["forwardingTimeouts"]
    assert {"dialTimeout", "responseHeaderTimeout", "idleConnTimeout"} <= set(timeouts)


def test_middleware_order_rejects_cheaply_and_retries_closest_to_service(
    dynamic: dict[str, Any],
) -> None:
    order = dynamic["http"]["routers"]["api"]["middlewares"]
    assert order.index("rate-limit") < order.index("circuit-breaker") < order.index("retry")
    assert order[-1] == "retry"


def test_ready_is_not_routed_at_the_edge(dynamic: dict[str, Any]) -> None:
    for router in dynamic["http"]["routers"].values():
        assert "/ready" not in router["rule"]
