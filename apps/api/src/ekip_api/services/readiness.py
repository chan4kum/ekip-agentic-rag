"""Readiness: can this instance serve traffic right now? (fail closed)"""

import asyncio
from collections.abc import Sequence

from ekip_contracts.health import DependencyProbe
from ekip_observability.logging import get_logger

log = get_logger("ekip.readiness")


class ReadinessService:
    def __init__(self, probes: Sequence[DependencyProbe], timeout_s: float = 2.0) -> None:
        if not probes:
            raise ValueError(
                "readiness needs at least one probe (an empty check is vacuously ready)"
            )
        self._probes = list(probes)
        self._timeout_s = timeout_s

    async def _run_one(self, probe: DependencyProbe) -> bool:
        try:
            async with asyncio.timeout(self._timeout_s):
                await probe.check()
        except Exception as exc:
            # Any failure or timeout means not ready. Log the type only: messages can
            # contain hostnames or DSN fragments (I-8).
            log.warning("probe_failed", probe=probe.name, error_type=type(exc).__name__)
            return False
        return True

    async def run(self) -> dict[str, bool]:
        results = await asyncio.gather(*(self._run_one(p) for p in self._probes))
        return {p.name: ok for p, ok in zip(self._probes, results, strict=True)}
