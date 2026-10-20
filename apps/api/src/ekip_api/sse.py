"""Server-Sent Events helpers: framing, heartbeats, an overall deadline."""

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import suppress
from typing import Any, Literal

# Event names are a closed set chosen by the server; user text only ever travels inside JSON.
EventName = Literal["meta", "token", "citations", "status", "done", "error"]

HEARTBEAT = b": keep-alive\n\n"  # an SSE comment: ignored by clients, keeps proxies from idling out


class StreamDeadlineError(Exception):
    """The stream ran longer than its overall deadline."""


def format_event(event: EventName, data: dict[str, Any]) -> bytes:
    # json.dumps never emits a raw newline (they become \n escapes), so user-controlled text
    # cannot end the data line and smuggle in a fake "event:" field.
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode()


async def with_heartbeats[T](
    source: AsyncIterator[T], *, interval_s: float, total_timeout_s: float
) -> AsyncGenerator[T | bytes]:
    """Re-yield `source`, inserting HEARTBEAT whenever it is quiet for `interval_s`.

    Raises StreamDeadlineError after `total_timeout_s`. The source runs in its own task fed through
    a small bounded queue (backpressure). When the consumer goes away (client disconnect ->
    cancellation) or finishes, the source task is cancelled and awaited, so its `finally` blocks
    (releasing connections, stopping upstream work) always run.
    """
    queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=16)

    async def pump() -> None:
        try:
            async for item in source:
                await queue.put(("item", item))
            await queue.put(("end", None))
        except Exception as exc:  # delivered to the consumer; CancelledError is not an Exception
            await queue.put(("error", exc))

    loop = asyncio.get_running_loop()
    deadline = loop.time() + total_timeout_s
    task = asyncio.create_task(pump())
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise StreamDeadlineError
            try:
                kind, value = await asyncio.wait_for(queue.get(), min(interval_s, remaining))
            except TimeoutError:
                if deadline - loop.time() <= 0:
                    raise StreamDeadlineError from None
                yield HEARTBEAT
                continue
            if kind == "item":
                yield value
            elif kind == "end":
                return
            else:
                raise value
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(task, return_exceptions=True)
