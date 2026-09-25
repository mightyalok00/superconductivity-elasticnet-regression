from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx


def request_once(url: str, timeout: float) -> tuple[int, float]:
    started = time.perf_counter()
    try:
        response = httpx.get(url, timeout=timeout)
        return response.status_code, (time.perf_counter() - started) * 1000
    except Exception:
        return 0, (time.perf_counter() - started) * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight HTTP load/smoke test.")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000/health",
        help="Endpoint to test.",
    )
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    latencies = []
    statuses = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(request_once, args.url, args.timeout)
            for _ in range(args.requests)
        ]
        for future in as_completed(futures):
            status, latency = future.result()
            statuses.append(status)
            latencies.append(latency)

    successful = sum(200 <= code < 400 for code in statuses)
    latencies_sorted = sorted(latencies)
    p95_index = max(0, min(len(latencies_sorted) - 1, int(len(latencies_sorted) * 0.95) - 1))

    print(f"URL: {args.url}")
    print(f"Requests: {args.requests}")
    print(f"Concurrency: {args.concurrency}")
    print(f"Successful: {successful}/{args.requests}")
    print(f"Mean latency: {statistics.mean(latencies):.2f} ms")
    print(f"Median latency: {statistics.median(latencies):.2f} ms")
    print(f"P95 latency: {latencies_sorted[p95_index]:.2f} ms")

    if successful != args.requests:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
