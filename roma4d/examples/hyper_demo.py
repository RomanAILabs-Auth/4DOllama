#!/usr/bin/env python3
"""Demo script for Roma4D HyperEngine (Phase 1): plain Python; try `r4d --explain hyper_demo.py`."""

from __future__ import annotations


def transform(x: float) -> float:
    return x * x + 1.0


def main() -> None:
    # Hot loop shape HyperEngine flags as a parallel candidate (advisory only).
    acc = 0.0
    for i in range(50_000):
        acc += transform(float(i))
    # Kinetic Bridge (Phase 2): numeric-only range loop — liftable to Roma4D serial kernel.
    k_acc = 0.0
    for k_i in range(10_000):
        k_acc = k_acc + float(k_i) * float(k_i)
    particles = [float(j) for j in range(100)]
    s = sum(particles)
    print(
        f"hyper_demo: acc={acc:.2f} k_acc={k_acc:.2f} sum={s:.2f} "
        f"(r4d --explain shows Kinetic kernel; R4D_KINETIC_TRY_COMPILE=1 to trial native build)"
    )


if __name__ == "__main__":
    main()
