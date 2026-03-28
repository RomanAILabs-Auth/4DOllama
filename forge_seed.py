#!/usr/bin/env python3
"""
forge_seed.py — Automate creation of a lightweight Cl(4,0)-shaped seed manifold
as safetensors payload (e.g. romanai_v2_part1.4dai for Modelfile FROM).

Usage:
  python forge_seed.py                    # -> romanai.4dai (or ROMANAI_FORGE_OUT)
  python forge_seed.py romanai_v2_part1.4dai
"""
from __future__ import annotations

import os
import sys

import torch
from safetensors.torch import save_file

SEED = int(os.environ.get("ROMANAI_FORGE_SEED", "42"))
OUT = os.environ.get("ROMANAI_FORGE_OUT", "romanai.4dai")
if len(sys.argv) > 1:
    OUT = sys.argv[1]
NUM_TENSORS = 10
SHAPE = (128, 576, 4, 4)


def main() -> None:
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    state: dict[str, torch.Tensor] = {}
    for i in range(NUM_TENSORS):
        # Foundational Cl(4,0) carrier blocks: [batch-like, fan-in, 4, 4] per RomanAI tensor layout.
        t = torch.randn(SHAPE, dtype=torch.float16)
        state[f"cl40_seed_{i}"] = t.contiguous()

    save_file(state, OUT)
    nbytes = sum(v.numel() * v.element_size() for v in state.values())
    print(f"Forge OK: {OUT} ({len(state)} tensors, {nbytes / 1e6:.2f} MB raw weights)")


if __name__ == "__main__":
    main()
