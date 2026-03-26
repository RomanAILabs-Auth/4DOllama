# engine.py
# Copyright RomanAILabs - Daniel Harding
# Christ is King.

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

from fourdollama.config import Settings
from fourdollama.kernel import write_kernel_r4d
from fourdollama.r4d_subprocess import stream_r4d_lines
from fourdollama.registry import ensure_model_registered


def ensure_model(settings: Settings, name: str) -> str:
    return ensure_model_registered(settings, name)


async def stream_engine(
    model: str,
    prompt: str,
    system: str | None = None,
    *,
    settings: Settings | None = None,
) -> AsyncIterator[str]:
    s = settings or Settings.load()
    canonical = ensure_model_registered(s, model)
    root = s.data_dir / "work"
    root.mkdir(parents=True, exist_ok=True)
    job = uuid.uuid4().hex
    d = root / job
    d.mkdir(parents=True, exist_ok=True)
    kpath = d / "kernel.r4d"
    write_kernel_r4d(kpath, canonical, prompt, system)
    async for line in stream_r4d_lines(s, kpath, d):
        yield line
