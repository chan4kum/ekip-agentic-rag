"""Infra guard: catches risky edits to docker-compose.yml without needing Docker."""

from pathlib import Path
from typing import Any

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


@pytest.fixture(scope="module")
def services() -> dict[str, Any]:
    data = yaml.safe_load(COMPOSE.read_text())
    services: dict[str, Any] = data["services"]
    return services


def test_required_services_exist(services: dict[str, Any]) -> None:
    assert {"postgres", "redis", "opensearch"} <= set(services)


def test_every_published_port_binds_to_localhost(services: dict[str, Any]) -> None:
    for name, svc in services.items():
        for port in svc.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name} exposes {port} beyond localhost"


def test_every_service_has_a_healthcheck(services: dict[str, Any]) -> None:
    for name, svc in services.items():
        assert "healthcheck" in svc, f"{name} has no healthcheck"


def test_no_floating_latest_tags(services: dict[str, Any]) -> None:
    for name, svc in services.items():
        if "image" not in svc:  # locally built service
            assert "build" in svc, f"{name} has neither image nor build"
            continue
        image = str(svc["image"])
        assert ":" in image, f"{name} image has no tag"
        assert not image.endswith(":latest"), f"{name} uses :latest"


def test_api_container_is_hardened(services: dict[str, Any]) -> None:
    api = services["api"]
    assert api["read_only"] is True
    assert api["cap_drop"] == ["ALL"]
    assert "no-new-privileges:true" in api["security_opt"]


def test_api_waits_for_healthy_dependencies(services: dict[str, Any]) -> None:
    deps = services["api"]["depends_on"]
    for name in ("postgres", "redis", "opensearch"):
        assert deps[name]["condition"] == "service_healthy"


def test_api_does_not_receive_the_whole_env_file(services: dict[str, Any]) -> None:
    assert "env_file" not in services["api"]


ROOT = COMPOSE.parent
DOCKERFILE = (ROOT / "apps/api/Dockerfile").read_text()


def test_dockerfile_runs_as_non_root() -> None:
    user_lines = [line for line in DOCKERFILE.splitlines() if line.startswith("USER ")]
    assert user_lines
    assert user_lines[-1].split()[1] not in {"root", "0"}


def test_dockerfile_bases_are_not_latest() -> None:
    froms = [line for line in DOCKERFILE.splitlines() if line.startswith("FROM ")]
    assert froms
    assert all(":latest" not in line and ":" in line.split()[1] for line in froms)


def test_dockerfile_installs_from_the_frozen_lockfile() -> None:
    assert "--frozen" in DOCKERFILE


def test_dockerignore_keeps_secrets_out_of_the_image() -> None:
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    assert ".env" in lines
    assert ".env.*" in lines
