import asyncio
from collections.abc import Iterator

import httpx
import pytest
import structlog
from ekip_api.main import create_app
from ekip_api.schemas import ReadyResponse
from ekip_api.services.readiness import ReadinessService
from ekip_config.settings import Settings

LEAKY_ERROR_TEXT = "postgres://user:hunter2@internal-db:5432/x"


class FakeProbe:
    def __init__(self, name: str, *, error: Exception | None = None, hang: bool = False) -> None:
        self._name = name
        self._error = error
        self._hang = hang

    @property
    def name(self) -> str:
        return self._name

    async def check(self) -> None:
        if self._hang:
            await asyncio.sleep(30)
        if self._error:
            raise self._error


def app_client(*probes: FakeProbe) -> httpx.AsyncClient:
    settings = Settings(_env_file=None, app_env="test")
    app = create_app(settings, probes=list(probes))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest.fixture(autouse=True)
def _reset_logging() -> Iterator[None]:
    structlog.reset_defaults()
    yield
    structlog.reset_defaults()


async def test_all_probes_ok_returns_200() -> None:
    async with app_client(FakeProbe("postgres"), FakeProbe("redis")) as c:
        r = await c.get("/ready")
    assert r.status_code == 200
    body = ReadyResponse.model_validate(r.json())
    assert body.status == "ready"
    assert body.checks == {"postgres": "ok", "redis": "ok"}


async def test_one_failing_probe_returns_503_and_marks_it() -> None:
    async with app_client(FakeProbe("postgres"), FakeProbe("redis", error=RuntimeError("x"))) as c:
        r = await c.get("/ready")
    assert r.status_code == 503
    body = ReadyResponse.model_validate(r.json())
    assert body.status == "not_ready"
    assert body.checks == {"postgres": "ok", "redis": "fail"}


async def test_hanging_probe_times_out_instead_of_hanging_the_endpoint() -> None:
    service = ReadinessService([FakeProbe("slow", hang=True), FakeProbe("fast")], timeout_s=0.05)
    result = await asyncio.wait_for(service.run(), timeout=2)
    assert result == {"slow": False, "fast": True}


async def test_probes_run_concurrently() -> None:
    service = ReadinessService([FakeProbe(f"s{i}", hang=True) for i in range(5)], timeout_s=0.2)
    loop = asyncio.get_running_loop()
    start = loop.time()
    await service.run()
    assert loop.time() - start < 0.6  # ~0.2s if concurrent, ~1.0s if sequential


async def test_error_text_never_reaches_the_response_and_only_type_is_logged(
    capsys: pytest.CaptureFixture[str],
) -> None:
    async with app_client(FakeProbe("postgres", error=ConnectionError(LEAKY_ERROR_TEXT))) as c:
        r = await c.get("/ready")
    assert "hunter2" not in r.text and "internal-db" not in r.text
    logs = capsys.readouterr().out
    assert "ConnectionError" in logs
    assert "hunter2" not in logs


def test_empty_probe_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        ReadinessService([])


def test_default_probes_cover_the_three_dependencies() -> None:
    from ekip_api.adapters.probes import build_default_probes

    names = [p.name for p in build_default_probes(Settings(_env_file=None, app_env="test"))]
    assert names == ["postgres", "redis", "opensearch"]


def test_asyncpg_probe_uses_a_plain_postgres_url() -> None:
    from ekip_api.adapters.probes import PostgresProbe, build_default_probes

    pg = build_default_probes(Settings(_env_file=None, app_env="test"))[0]
    assert isinstance(pg, PostgresProbe)
    assert "+asyncpg" not in pg._dsn
