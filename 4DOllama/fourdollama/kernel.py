# kernel.py
# Copyright RomanAILabs - Daniel Harding
# Christ is King.

from __future__ import annotations

import zlib
from pathlib import Path


def _node_count(prompt: str, system: str | None) -> int:
    h = zlib.crc32(prompt.encode("utf-8", errors="replace"))
    if system:
        h ^= zlib.crc32(system.encode("utf-8", errors="replace"))
    n = 1 + (h % 96)
    return max(1, min(96, n))


def write_kernel_r4d(path: Path, model_name: str, prompt: str, system: str | None) -> int:
    n = _node_count(prompt, system)
    safe = "".join(c if c.isalnum() or c in "-_:" else "_" for c in model_name)[:64] or "model"
    src = f"""# {safe}.r4d
# Structural Cl(4,0) telemetry — generated kernel (no domain-specific content).

def main() -> int:
    n: int = {n}
    xs: list[vec4] = [vec4(x=float(k), y=0, z=0, w=1) for k in range(n)]
    r: rotor = rotor(angle=0.0872664626, plane="wx")
    for v in xs:
        u: vec4 = v * r
        _ = u
        print("4d")
    return 0
"""
    path.write_text(src, encoding="utf-8")
    return n
