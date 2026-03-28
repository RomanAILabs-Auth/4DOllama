#!/usr/bin/env python3
"""
bench_4d.py — scalar hot loop for wall-clock compare with Bench_4d.r4d and bench_4d.rs.

Run:  python examples/bench_4d.py
Windows (r4d + python + rust):  examples/run_bench_4d.ps1 from repo root.
Rust GNU on Windows:  rustup target add x86_64-pc-windows-gnu
"""
from __future__ import annotations

import time

N = 5_000_000


def main() -> None:
    t0 = time.perf_counter()
    s = 0.0
    for i in range(N):
        x = i * 1e-7
        s += x * x + (1.0 - x) * 0.5
    dt = time.perf_counter() - t0
    print(f"bench_4d python: N={N} sum={s:.6f} wall_s={dt:.4f}")


if __name__ == "__main__":
    main()
