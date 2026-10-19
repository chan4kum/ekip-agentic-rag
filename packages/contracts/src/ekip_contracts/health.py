"""Port for checking that an infrastructure dependency is usable."""

from typing import Protocol


class DependencyProbe(Protocol):
    """One health check against one dependency (database, cache, search...)."""

    @property
    def name(self) -> str:
        """Short stable label used in readiness output, e.g. 'postgres'."""
        ...

    async def check(self) -> None:
        """Return normally if healthy; raise any exception if not."""
        ...
