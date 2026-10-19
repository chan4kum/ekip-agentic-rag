"""Live checks against the Compose stack. Run: RUN_INTEGRATION=1 uv run pytest tests/integration"""

import os

import httpx
import pytest
from ekip_api.adapters.probes import RedisProbe, build_default_probes
from ekip_api.main import create_app
from ekip_config.settings import Settings
from fastapi import FastAPI

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="needs live Compose stack"),
]


def client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_ready_is_200_when_all_services_are_up() -> None:
    app = create_app(Settings(_env_file=None, app_env="test"))
    async with client(app) as c:
        r = await c.get("/ready")
    assert r.status_code == 200, r.text
    assert r.json()["checks"] == {"postgres": "ok", "redis": "ok", "opensearch": "ok"}


async def test_ready_is_503_when_redis_is_unreachable() -> None:
    settings = Settings(_env_file=None, app_env="test")
    probes = [p for p in build_default_probes(settings) if p.name != "redis"]
    probes.append(RedisProbe("redis://127.0.0.1:1/0", timeout_s=0.5))
    async with client(create_app(settings, probes=probes)) as c:
        r = await c.get("/ready")
    assert r.status_code == 503
    assert r.json()["checks"]["redis"] == "fail"
