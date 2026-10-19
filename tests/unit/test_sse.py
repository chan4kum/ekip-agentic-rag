import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import aclosing

import pytest
from ekip_api.sse import HEARTBEAT, StreamDeadlineError, format_event, with_heartbeats


def test_event_framing() -> None:
    assert format_event("token", {"text": "hi"}) == b'event: token\ndata: {"text":"hi"}\n\n'


def test_user_text_cannot_inject_events_or_break_framing() -> None:
    evil = "x\n\nevent: done\ndata: {}\n\r\n"
    raw = format_event("token", {"text": evil})
    # exactly one event: one 'event:' line, one 'data:' line, one blank-line terminator
    assert raw.count(b"\n\n") == 1 and raw.endswith(b"\n\n")
    lines = raw.decode().split("\n")
    assert lines[0] == "event: token" and lines[1].startswith("data: ")
    assert json.loads(lines[1][6:])["text"] == evil  # round-trips intact inside the JSON string


def test_non_ascii_is_kept_readable() -> None:
    assert "héllo".encode() in format_event("token", {"text": "héllo"})


async def source(*items: int, delay: float = 0.0) -> AsyncIterator[int]:
    for i in items:
        if delay:
            await asyncio.sleep(delay)
        yield i


async def collect(gen: AsyncIterator[object]) -> list[object]:
    return [x async for x in gen]


async def test_items_pass_through_in_order() -> None:
    out = await collect(with_heartbeats(source(1, 2, 3), interval_s=1, total_timeout_s=5))
    assert out == [1, 2, 3]


async def test_quiet_source_gets_heartbeats() -> None:
    out = await collect(with_heartbeats(source(1, delay=0.25), interval_s=0.05, total_timeout_s=5))
    assert out[-1] == 1 and out.count(HEARTBEAT) >= 3


async def test_source_errors_propagate_to_the_consumer() -> None:
    async def failing() -> AsyncIterator[int]:
        yield 1
        raise RuntimeError("boom")

    seen: list[object] = []
    with pytest.raises(RuntimeError, match="boom"):
        async for item in with_heartbeats(failing(), interval_s=1, total_timeout_s=5):
            seen.append(item)
    assert seen == [1]


async def test_overall_deadline_raises() -> None:
    with pytest.raises(StreamDeadlineError):
        await collect(with_heartbeats(source(1, delay=5), interval_s=0.05, total_timeout_s=0.2))


async def test_source_is_cleaned_up_when_the_consumer_stops_early() -> None:
    state = {"closed": False}

    async def endless() -> AsyncIterator[int]:
        try:
            while True:
                yield 1
                await asyncio.sleep(0.01)
        finally:
            state["closed"] = True

    async with aclosing(with_heartbeats(endless(), interval_s=1, total_timeout_s=5)) as gen:
        assert await anext(gen) == 1
    assert state["closed"] is True


async def test_source_is_cleaned_up_on_cancellation() -> None:
    state = {"closed": False}

    async def hang() -> AsyncIterator[int]:
        try:
            await asyncio.sleep(10)
            yield 1
        finally:
            state["closed"] = True

    task = asyncio.create_task(collect(with_heartbeats(hang(), interval_s=1, total_timeout_s=30)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert state["closed"] is True


async def test_a_slow_consumer_applies_backpressure_instead_of_buffering_everything() -> None:
    produced = 0

    async def fast() -> AsyncIterator[int]:
        nonlocal produced
        for i in range(1000):
            produced += 1
            yield i

    gen = with_heartbeats(fast(), interval_s=1, total_timeout_s=5)
    await anext(gen)
    await asyncio.sleep(0.1)
    assert produced < 100  # bounded queue: the producer stalls instead of racing ahead
    await gen.aclose()
