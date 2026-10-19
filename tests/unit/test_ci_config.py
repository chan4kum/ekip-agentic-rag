"""Guards for CI, pinned images and architecture rules (no Docker or network needed)."""

import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
LINT_IMPORTS = str(Path(sys.executable).parent / "lint-imports")


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
    return data


def all_steps(workflow: dict[str, Any]) -> list[dict[str, Any]]:
    return [step for job in workflow["jobs"].values() for step in job["steps"]]


def test_workflow_token_is_read_only(workflow: dict[str, Any]) -> None:
    assert workflow["permissions"] == {"contents": "read"}


def test_every_action_is_pinned_to_a_commit_sha(workflow: dict[str, Any]) -> None:
    uses = [s["uses"] for s in all_steps(workflow) if "uses" in s]
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", ref), f"not SHA-pinned: {ref}"


def test_checkout_does_not_persist_credentials(workflow: dict[str, Any]) -> None:
    for step in all_steps(workflow):
        if str(step.get("uses", "")).startswith("actions/checkout"):
            assert step["with"]["persist-credentials"] is False


def test_quality_job_runs_all_gates(workflow: dict[str, Any]) -> None:
    commands = " ".join(s.get("run", "") for s in workflow["jobs"]["quality"]["steps"])
    for needle in (
        "uv sync --frozen",
        "ruff check",
        "ruff format --check",
        "mypy",
        "lint-imports",
        "pytest",
    ):
        assert needle in commands, f"quality job misses: {needle}"


def test_stack_job_runs_the_live_smoke_test_and_always_cleans_up(workflow: dict[str, Any]) -> None:
    job = workflow["jobs"]["stack"]
    assert job["needs"] == "quality"
    commands = " ".join(s.get("run", "") for s in job["steps"])
    assert "docker compose up" in commands and "--wait" in commands
    assert "scripts/smoke.sh" in commands
    assert "RUN_INTEGRATION=1" in commands
    assert job["steps"][-1]["if"] == "always()"
    assert "down -v" in job["steps"][-1]["run"]


def test_workflow_uses_no_secrets(workflow: dict[str, Any]) -> None:
    assert "secrets." not in (ROOT / ".github/workflows/ci.yml").read_text()


def test_smoke_script_fails_fast() -> None:
    text = (ROOT / "scripts/smoke.sh").read_text()
    assert "set -euo pipefail" in text
    assert text.startswith("#!/usr/bin/env bash")


def test_every_compose_image_is_pinned_by_digest() -> None:
    compose = yaml.safe_load((ROOT / "docker-compose.yml").read_text())
    for name, svc in compose["services"].items():
        image = svc.get("image")
        if image:  # api is built locally
            assert re.search(r"@sha256:[0-9a-f]{64}$", image), f"{name} image not digest-pinned"


def test_dependabot_covers_actions_docker_and_uv() -> None:
    cfg = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text())
    ecosystems = {u["package-ecosystem"] for u in cfg["updates"]}
    assert {"github-actions", "docker", "uv"} <= ecosystems


def test_architecture_rules_hold_on_the_real_repo() -> None:
    result = subprocess.run(  # noqa: S603 (fixed argv, no shell)
        [LINT_IMPORTS], cwd=ROOT, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_architecture_rules_actually_fail_on_a_violation(tmp_path: Path) -> None:
    pkg = tmp_path / "toypkg"
    (pkg / "domain").mkdir(parents=True)
    (pkg / "infra").mkdir()
    for init in (pkg / "__init__.py", pkg / "domain/__init__.py", pkg / "infra/__init__.py"):
        init.write_text("")
    (pkg / "infra/db.py").write_text("VALUE = 1\n")
    (pkg / "domain/service.py").write_text("from toypkg.infra import db  # forbidden\n")
    (tmp_path / ".importlinter").write_text(
        "[importlinter]\nroot_package = toypkg\n\n"
        "[importlinter:contract:1]\nname = domain never imports infra\n"
        "type = forbidden\nsource_modules = toypkg.domain\nforbidden_modules = toypkg.infra\n"
    )
    result = subprocess.run(  # noqa: S603
        [LINT_IMPORTS], cwd=tmp_path, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert "BROKEN" in result.stdout
