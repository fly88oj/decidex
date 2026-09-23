"""Concurrency load test against a running Decidex server.

Verifies the stability work: the event loop stays responsive under GPU-bound
load, overload produces 429 + Retry-After (never 500s), and retries succeed.

    python tests/load_test.py [--concurrency 32] [--base-url http://127.0.0.1:8600]
"""

from __future__ import annotations

import argparse
import time
from collections import Counter

import httpx

REQUEST_TEMPLATE = {
    "state": "Customer says: {i} I was charged twice for order A-10{i} and want a refund now!",
    "model": "decidex-latest",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
                "sales": "Pricing, upgrades, new accounts",
            },
        },
        "urgent": {"type": "noul", "instructions": "Does this convey urgency?"},
    },
}


def one_request(client: httpx.Client, i: int, max_attempts: int = 4) -> tuple[int, float, bool]:
    body = {
        "state": REQUEST_TEMPLATE["state"].format(i=i % 8),
        "model": "decidex-latest",
        "questions": REQUEST_TEMPLATE["questions"],
    }
    for attempt in range(max_attempts):
        t0 = time.perf_counter()
        response = client.post("/v1/systemone", json=body)
        elapsed = (time.perf_counter() - t0) * 1000
        if response.status_code == 200:
            return 200, elapsed, response.json()["answers"]["department"]["choice"] == "billing"
        if response.status_code == 429:
            time.sleep(float(response.headers.get("retry-after", "1")))
            continue
        return response.status_code, elapsed, False
    return 429, 0.0, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--base-url", default="http://127.0.0.1:8600")
    args = parser.parse_args()

    import threading

    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        # sanity single request + warmup
        status, ms, ok = one_request(client, 0, max_attempts=1)
        assert status == 200 and ok, f"sanity request failed: {status}"

        # health responsiveness during load: measure /health latency while
        # workers hammer /v1/systemone
        health_during_load: list[float] = []
        stop = threading.Event()

        def health_probe() -> None:
            while not stop.is_set():
                t0 = time.perf_counter()
                client.get("/health", timeout=10)
                health_during_load.append((time.perf_counter() - t0) * 1000)
                time.sleep(0.1)

        probe = threading.Thread(target=health_probe, daemon=True)
        probe.start()

        statuses: Counter = Counter()
        latencies: list[float] = []
        correct_flags: list[bool] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            status, ms, ok = one_request(client, i)
            with lock:
                statuses[status] += 1
                if status == 200:
                    latencies.append(ms)
                    correct_flags.append(ok)

        t0 = time.perf_counter()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(args.concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_s = time.perf_counter() - t0
        stop.set()
        probe.join(timeout=2)

    latencies.sort()
    print(f"concurrency={args.concurrency}  wall={total_s:.1f}s")
    print(f"statuses: {dict(statuses)}")
    if latencies:
        print(f"200 latency ms: p50={latencies[len(latencies)//2]:.0f} "
              f"p95={latencies[int(len(latencies)*0.95)]:.0f} max={latencies[-1]:.0f}")
        print(f"correct answers: {sum(correct_flags)}/{len(correct_flags)}")
    if health_during_load:
        health_during_load.sort()
        print(f"/health during load ms: p50={health_during_load[len(health_during_load)//2]:.0f} "
              f"max={health_during_load[-1]:.0f}")

    ok = statuses[500] == 0 and statuses[200] == args.concurrency
    print("LOAD TEST:", "PASS" if ok else "FAIL",
          "(no 5xx, all requests eventually 200 via retry)")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
