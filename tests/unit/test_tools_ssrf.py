import pytest
from ekip_tools import OutboundBlockedError, assert_public_host, check_outbound_url
from ekip_tools.ssrf import is_public_ip

ALLOWED = ["api.example.com"]


def test_allowlisted_https_host_passes() -> None:
    assert check_outbound_url("https://API.example.com/v1/search?q=1", ALLOWED) == "api.example.com"
    assert check_outbound_url("https://api.example.com./x", ALLOWED) == "api.example.com"


@pytest.mark.parametrize(
    ("url", "reason"),
    [
        ("http://api.example.com/x", "scheme_not_https"),
        ("file:///etc/passwd", "scheme_not_https"),
        ("https://user:pw@api.example.com/x", "credentials_in_url"),
        ("https://api.example.com:8443/x", "port_not_allowed"),
        ("https://127.0.0.1/x", "ip_literal"),
        ("https://[::1]/x", "ip_literal"),
        ("https://169.254.169.254/latest/meta-data", "ip_literal"),
        ("https://2130706433/x", "host_not_allowlisted"),
        ("https://evil.example.com/x", "host_not_allowlisted"),
        ("https://api.example.com.evil.io/x", "host_not_allowlisted"),
        ("https:///x", "no_host"),
    ],
)
def test_dangerous_urls_are_blocked(url: str, reason: str) -> None:
    with pytest.raises(OutboundBlockedError) as err:
        check_outbound_url(url, ALLOWED)
    assert err.value.reason == reason


@pytest.mark.parametrize(
    ("address", "public"),
    [
        ("93.184.216.34", True),
        ("10.0.0.5", False),
        ("192.168.1.1", False),
        ("127.0.0.1", False),
        ("169.254.169.254", False),
        ("::1", False),
        ("fd00::1", False),
        ("::ffff:10.0.0.1", False),
        ("224.0.0.1", False),
        ("not-an-ip", False),
    ],
)
def test_public_ip_classification(address: str, public: bool) -> None:
    assert is_public_ip(address) is public


async def test_names_resolving_to_internal_addresses_are_blocked() -> None:
    async def internal(_: str) -> list[str]:
        return ["93.184.216.34", "10.1.2.3"]

    async def public(_: str) -> list[str]:
        return ["93.184.216.34"]

    async def broken(_: str) -> list[str]:
        raise OSError("nxdomain")

    with pytest.raises(OutboundBlockedError, match="non_public_address"):
        await assert_public_host("api.example.com", internal)
    await assert_public_host("api.example.com", public)
    with pytest.raises(OutboundBlockedError, match="unresolvable"):
        await assert_public_host("api.example.com", broken)
