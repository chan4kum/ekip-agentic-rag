"""Guards for the egress (forward) proxy and the network layout (no Docker needed)."""

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONF = (ROOT / "infrastructure/squid/squid.conf").read_text()
LINES = [ln.strip() for ln in CONF.splitlines() if ln.strip() and not ln.strip().startswith("#")]


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    return data


def http_access() -> list[str]:
    return [ln for ln in LINES if ln.startswith("http_access ")]


def test_last_rule_denies_everything() -> None:
    assert http_access()[-1] == "http_access deny all"


def test_only_one_allow_rule_and_it_requires_source_and_host_acls() -> None:
    allows = [ln for ln in http_access() if ln.startswith("http_access allow")]
    assert allows == ["http_access allow app_net allowed_hosts"]


def test_private_ranges_are_denied_before_the_allow_rule() -> None:
    rules = http_access()
    deny = rules.index("http_access deny private_dst")
    allow = rules.index("http_access allow app_net allowed_hosts")
    assert deny < allow


def test_only_https_connect_is_permitted() -> None:
    rules = http_access()
    assert "http_access deny !https_port" in rules
    assert "http_access deny !CONNECT" in rules
    assert "acl https_port port 443" in LINES


def test_no_tls_interception_and_no_caching() -> None:
    assert not any(ln.startswith(("ssl_bump", "https_port", "sslcrtd")) for ln in LINES)
    assert "cache deny all" in LINES


def test_allowlist_has_exact_hostnames_only() -> None:
    hosts = [ln.split()[-1] for ln in LINES if ln.startswith("acl allowed_hosts dstdomain ")]
    assert hosts, "allowlist must not be empty"
    for host in hosts:
        assert re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", host), f"not an exact host: {host}"


def test_private_ranges_cover_metadata_and_rfc1918() -> None:
    acl = next(ln for ln in LINES if ln.startswith("acl private_dst dst"))
    for cidr in ("169.254.0.0/16", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8"):
        assert cidr in acl


def test_proxy_source_acl_matches_the_app_subnet(compose: dict[str, Any]) -> None:
    subnet = compose["networks"]["app"]["ipam"]["config"][0]["subnet"]
    assert f"acl app_net src {subnet}" in LINES


def test_api_has_no_direct_internet_route(compose: dict[str, Any]) -> None:
    assert compose["services"]["api"]["networks"] == ["app"]
    assert compose["networks"]["app"]["internal"] is True


def test_only_squid_joins_the_egress_network(compose: dict[str, Any]) -> None:
    on_egress = [
        name for name, svc in compose["services"].items() if "egress" in (svc.get("networks") or [])
    ]
    assert on_egress == ["squid"]
    assert "internal" not in compose["networks"]["egress"]


def test_api_sends_https_through_squid_but_not_internal_services(compose: dict[str, Any]) -> None:
    env = compose["services"]["api"]["environment"]
    assert env["HTTPS_PROXY"] == "http://squid:3128"
    for host in ("postgres", "redis", "opensearch"):
        assert host in env["NO_PROXY"].split(",")
    assert compose["services"]["api"]["depends_on"]["squid"]["condition"] == "service_healthy"


def test_squid_is_hardened(compose: dict[str, Any]) -> None:
    svc = compose["services"]["squid"]
    assert svc["user"] != "0" and not str(svc["user"]).startswith("0:")
    assert svc["read_only"] is True
    assert svc["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in svc["security_opt"]
    assert "ports" not in svc
    assert ":latest" not in svc["image"]
