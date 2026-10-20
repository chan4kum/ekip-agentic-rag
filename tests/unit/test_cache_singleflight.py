import asyncio

import pytest
from ekip_caching import SingleFlight


async def test_concurrent_calls_share_one_run() -> None:
    flight = SingleFlight()
    gate = asyncio.Event()
    runs = 0

    async def work() -> int:
        nonlocal runs
        runs += 1
        await gate.wait()
        return 42

    tasks = [asyncio.create_task(flight.run("k", work)) for _ in range(50)]
    await asyncio.sleep(0)
    gate.set()
    assert await asyncio.gather(*tasks) == [42] * 50 and runs == 1
    assert flight.in_flight == 0


async def test_different_keys_run_independently() -> None:
    flight = SingleFlight()

    async def work(n: int) -> int:
        await asyncio.sleep(0)
        return n

    results = await asyncio.gather(
        flight.run("a", lambda: work(1)), flight.run("b", lambda: work(2))
    )
    assert list(results) == [1, 2]


async def test_sequential_calls_each_run() -> None:
    flight = SingleFlight()
    runs = 0

    async def work() -> int:
        nonlocal runs
        runs += 1
        return runs

    assert [await flight.run("k", work), await flight.run("k", work)] == [1, 2]


async def test_error_reaches_every_waiter_and_clears_the_slot() -> None:
    flight = SingleFlight()
    gate = asyncio.Event()

    async def boom() -> int:
        await gate.wait()
        raise ValueError("bad")

    tasks = [asyncio.create_task(flight.run("k", boom)) for _ in range(3)]
    await asyncio.sleep(0)
    gate.set()
    for r in await asyncio.gather(*tasks, return_exceptions=True):
        assert isinstance(r, ValueError)
    assert flight.in_flight == 0

    async def ok() -> int:
        return 1

    assert await flight.run("k", ok) == 1


async def test_leader_cancellation_hands_over_to_a_waiter() -> None:
    flight = SingleFlight()
    started = asyncio.Event()
    runs: list[str] = []

    async def slow() -> str:
        runs.append("leader")
        started.set()
        await asyncio.sleep(60)
        return "never"

    async def fast() -> str:
        runs.append("follower")
        return "done"

    leader = asyncio.create_task(flight.run("k", slow))
    await started.wait()
    follower = asyncio.create_task(flight.run("k", fast))
    await asyncio.sleep(0)
    leader.cancel()
    with pytest.raises(asyncio.CancelledError):
        await leader
    assert await follower == "done" and runs == ["leader", "follower"]


async def test_cancelled_waiter_does_not_cancel_the_leader() -> None:
    flight = SingleFlight()
    gate = asyncio.Event()

    async def work() -> int:
        await gate.wait()
        return 7

    leader = asyncio.create_task(flight.run("k", work))
    await asyncio.sleep(0)
    waiter = asyncio.create_task(flight.run("k", work))
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    gate.set()
    assert await leader == 7
