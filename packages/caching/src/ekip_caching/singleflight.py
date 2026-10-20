"""Single-flight: concurrent identical misses run ONE computation (per process).

Callers are keyed by the full cache key, which already binds tenant and caller identity, so
nobody ever receives a result computed for someone else. If the leader is cancelled, waiting
callers do not fail: one of them becomes the new leader. Across processes the worst case is one
computation per process, which is acceptable; a distributed lock is deliberately not added.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any


class _LeaderCancelled(Exception):  # noqa: N818 - internal control-flow signal
    pass


class SingleFlight:
    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}

    @property
    def in_flight(self) -> int:
        return len(self._inflight)

    async def run[T](self, key: str, fn: Callable[[], Awaitable[T]]) -> T:
        while (fut := self._inflight.get(key)) is not None:
            try:
                result: T = await asyncio.shield(fut)
            except _LeaderCancelled:
                continue
            return result
        fut = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            result = await fn()
        except asyncio.CancelledError:
            fut.set_exception(_LeaderCancelled())
            fut.exception()  # mark retrieved: there may be no waiters
            raise
        except BaseException as exc:
            fut.set_exception(exc)
            fut.exception()
            raise
        else:
            fut.set_result(result)
            return result
        finally:
            if self._inflight.get(key) is fut:
                del self._inflight[key]
