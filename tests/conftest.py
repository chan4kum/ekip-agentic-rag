from collections.abc import Iterator

import pytest
import structlog
from ekip_observability.logging import remove_owned_handlers


@pytest.fixture(autouse=True)
def _isolate_logging() -> Iterator[None]:
    """configure_logging() mutates global state; undo it so tests cannot affect each other."""
    yield
    remove_owned_handlers()
    structlog.reset_defaults()
