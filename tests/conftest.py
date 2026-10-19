import os
import subprocess
import sys
from collections.abc import Iterator

import pytest
import structlog
from ekip_observability.logging import remove_owned_handlers
from tests_paths import ROOT


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """configure_logging() mutates global state; undo it so tests cannot affect each other."""
    yield
    remove_owned_handlers()
    structlog.reset_defaults()


@pytest.fixture(scope="session")
def migrated() -> None:
    """Bring the live database to the latest schema (only when RUN_INTEGRATION=1)."""
    if os.environ.get("RUN_INTEGRATION") != "1":
        return
    url = os.environ.get("DATABASE_URL", "postgresql+asyncpg://ekip:ekip@localhost:5432/ekip")
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [
            sys.executable,
            "-m",
            "alembic",
            "-c",
            str(ROOT / "packages/persistence/alembic.ini"),
            "upgrade",
            "head",
        ],
        env={**os.environ, "DATABASE_URL": url},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
