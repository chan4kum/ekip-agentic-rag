"""Small asyncio load generator for the EKIP API (latency percentiles, error rate, SSE first token).

    export EKIP_LOAD_TOKEN=...   # an access token; never passed on the command line
    uv run --no-sync python scripts/load_test.py --base-url http://localhost:8000 \\
        --concurrency 8 --duration 30 --json-out load.json

It refuses any target that is not localhost unless --i-know is given: pointing a load generator at a
shared or production environment by accident is an outage you cause yourself. The JSON output uses
the flat `load.*` keys that scripts/eval_gate.py reads (`--runtime-results load.json`).
Results measure THIS client against THAT server only; they are not capacity numbers until the
test runs against production-like hardware and data.
"""

import argparse
import asyncio
import itertools
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import httpx
from ekip_observability.stats import percentile

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_MESSAGES = (
    "What is the landing gear inspection interval?",
    "Compare the previous and current maintenance procedures and list the changes.",
    "Who approves a deviation from the torque specification?",
    "Summarise the fatigue testing requirements.",
)


class TargetRefused(Exception):
    pass


def ensure_target_allowed(base_url: str, *, i_know: bool) -> None:
    host = urlparse(base_url).hostname or ""
    if host in LOCAL_HOSTS or i_know:
        return
    raise TargetRefused(
        f"refusing to load-test non-local host {host!r}; pass --i-know if this is intended"
    )


@dataclass
class LoadResult:
    started: float = 0.0
    finished: float = 0.0
    latencies_ms: list[float] = field(default_factory=list)
    ttft_ms: list[float] = field(default_factory=list)
    statuses: Counter[str] = field(default_factory=Counter)
    server_errors: int = 0
    rejected: int = 0  # 4xx, including 429
    requests: int = 0

    @property
    def duration_s(self) -> float:
        return max(self.finished - self.started, 1e-9)


def summarize(result: LoadResult) -> dict[str, float]:
    out: dict[str, float] = {
        "load.requests": float(result.requests),
        "load.rps": result.requests / result.duration_s,
        "load.error_rate": result.server_errors / result.requests if result.requests else 1.0,
        "load.rejected_rate": result.rejected / result.requests if result.requests else 0.0,
    }
    if result.latencies_ms:
        for q in (50, 95, 99):
            out[f"load.p{q}_ms"] = percentile(result.latencies_ms, q)
    if result.ttft_ms:
        for q in (50, 95):
            out[f"load.ttft_p{q}_ms"] = percentile(result.ttft_ms, q)
    return out


async def _one_request(
    client: httpx.AsyncClient,
    path: str,
    message: str,
    headers: dict[str, str],
    result: LoadResult,
    clock: Callable[[], float],
) -> None:
    started = clock()
    ttft: float | None = None
    sse_error = False
    status_label = "transport_error"
    server_error = False
    rejected = False
    try:
        async with client.stream("POST", path, json={"message": message}, headers=headers) as resp:
            status_label = str(resp.status_code)
            if resp.status_code >= 500:
                server_error = True
            elif resp.status_code >= 400:
                rejected = True
            else:
                buffer = ""
                async for chunk in resp.aiter_text():
                    buffer += chunk
                    if ttft is None and "event: token" in buffer:
                        ttft = (clock() - started) * 1000
                    if "event: error" in buffer:
                        sse_error = True
                    buffer = buffer[-64:]  # enough to span a chunk boundary
    except httpx.HTTPError:
        server_error = True
    elapsed_ms = (clock() - started) * 1000
    result.requests += 1
    result.statuses[status_label] += 1
    if server_error or sse_error:
        result.server_errors += 1
    elif rejected:
        result.rejected += 1
    else:
        result.latencies_ms.append(elapsed_ms)
        if ttft is not None:
            result.ttft_ms.append(ttft)


async def run_load(
    client: httpx.AsyncClient,
    *,
    path: str = "/api/v1/chat",
    messages: Sequence[str] = DEFAULT_MESSAGES,
    headers: dict[str, str] | None = None,
    concurrency: int = 4,
    duration_s: float | None = 10.0,
    max_requests: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> LoadResult:
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if duration_s is None and max_requests is None:
        raise ValueError("give a duration or a request count (otherwise the run never ends)")
    result = LoadResult(started=clock())
    cycle = itertools.cycle(messages)
    issued = 0
    deadline = None if duration_s is None else clock() + duration_s

    async def worker() -> None:
        nonlocal issued
        while True:
            if deadline is not None and clock() >= deadline:
                return
            if max_requests is not None and issued >= max_requests:
                return
            issued += 1  # single event loop: no await between check and increment
            await _one_request(client, path, next(cycle), headers or {}, result, clock)

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    result.finished = clock()
    return result


def format_summary(metrics: dict[str, float]) -> str:
    lines = [f"{k:22s} {v:10.3f}" for k, v in sorted(metrics.items())]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--path", default="/api/v1/chat")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--duration", type=float, default=10.0, help="seconds")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--messages-file", type=Path, help="one question per line")
    parser.add_argument("--token-env", default="EKIP_LOAD_TOKEN")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--i-know", action="store_true", help="allow a non-localhost target")
    args = parser.parse_args()
    try:
        ensure_target_allowed(args.base_url, i_know=args.i_know)
    except TargetRefused as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    token = os.environ.get(args.token_env)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    if not token:
        print(
            f"warning: ${args.token_env} is not set; requests go out unauthenticated",
            file=sys.stderr,
        )
    messages: Sequence[str] = DEFAULT_MESSAGES
    if args.messages_file:
        messages = [ln.strip() for ln in args.messages_file.read_text().splitlines() if ln.strip()]
        if not messages:
            print("error: messages file is empty", file=sys.stderr)
            return 2

    async def go() -> dict[str, float]:
        async with httpx.AsyncClient(
            base_url=args.base_url, timeout=args.timeout, trust_env=False
        ) as client:
            res = await run_load(
                client,
                path=args.path,
                messages=messages,
                headers=headers,
                concurrency=args.concurrency,
                duration_s=None if args.max_requests else args.duration,
                max_requests=args.max_requests,
            )
        return summarize(res)

    metrics = asyncio.run(go())
    print(format_summary(metrics))
    if args.json_out:
        args.json_out.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["load.error_rate"] < 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
